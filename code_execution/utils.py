"""Utility functions for code execution."""

import asyncio
import contextlib
import errno
import functools
import gc
import inspect
import io
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import structlog
from tqdm import tqdm

logger = structlog.get_logger()


# Taken from https://github.com/bytedance/SandboxFusion/blob/main/sandbox/utils/execution.py
T = TypeVar("T", bound=Callable[..., Coroutine[Any, Any, Any]])


def max_concurrency(limit: int) -> Callable[[T], T]:
    """Decorator to limit the maximum number of concurrent executions of an async function"""
    semaphore = asyncio.Semaphore(limit)

    def decorator(func: T) -> T:

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def in_notebook():
    """Checks if the code is running in a notebook."""
    try:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=import-error
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


class RunThread(threading.Thread):
    """Class that will allow asycnio to run in a thread when called from Jupyter."""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.had_error = False
        super().__init__()

    def run(self):
        try:
            self.result = asyncio.run(self.func(*self.args, **self.kwargs))
        except Exception as e:
            self.result = e
            self.had_error = True


def notebook_safe_async_run(target, *args, **kwargs):
    """Run an async function in a thread."""
    if in_notebook():
        logger.info("Running in separate thread due to notebook.")
        thread = RunThread(target, *args, **kwargs)
        thread.start()
        thread.join()
        if thread.had_error:
            raise thread.result
        return thread.result
    logger.info("Running in main thread.")
    return asyncio.run(target(*args, **kwargs))


def _batched_wrapper(batch, processor, proc_returns_list):
    """Wrapper for batched processing."""
    out = []
    for example in batch:
        idx = example["idx"]
        start = time.time()
        result = processor(*example["args"])
        elapsed = time.time() - start
        if proc_returns_list:
            out.extend([((idx, i), elapsed, r) for i, r in enumerate(result)])
        else:
            out.append(((idx, 0), elapsed, result))
    return out


def _normal_wrapper(arg_dict, processor, proc_returns_list):
    """Wrapper for normal processing."""
    start = time.time()
    result = processor(*arg_dict["args"])
    elapsed = time.time() - start
    if proc_returns_list:
        return [
            ((arg_dict["idx"], i), elapsed, r) for i, r in enumerate(result)
        ]
    return ((arg_dict["idx"], 0), elapsed, result)


def wrap_processor(
    processor_fn: Callable,
    batch_size: int,
    returns_list: bool,
) -> Callable:
    """Wraps a processor function to handle batching."""
    if batch_size > 1:
        logger.debug("Using batched processing with size %d", batch_size)
        wrapper = _batched_wrapper
    else:
        logger.debug("Using normal processing")
        wrapper = _normal_wrapper
    return functools.partial(
        wrapper, processor=processor_fn, proc_returns_list=returns_list
    )


def get_pred_dir(idx: int, parent: Path):
    """Gets the prediction directory for a prediction."""
    return parent.joinpath(f"pred{idx}")


class ContextTimeLimitException(Exception):
    """Timeout error for running commands."""


def timeout_signal_handler(signum, frame):
    raise ContextTimeLimitException(errno.ETIME)


# Timeout for windows.
class TimeoutContext:
    def __init__(self, seconds, on_end=None):
        self.seconds = seconds
        self.timer = None
        self.error_raised = False
        self.on_end = on_end

    def _timeout_handler(self):
        self.error_raised = True

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self._timeout_handler)
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer is not None:
            self.timer.cancel()
        if self.error_raised:
            raise ContextTimeLimitException(
                f"Operation timed out after {self.seconds} seconds"
            )


ON_WINDOWS = os.name == "nt"


def timeout_decorator(seconds: int = 10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ON_WINDOWS:
                signal.signal(signal.SIGALRM, timeout_signal_handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
            else:
                with TimeoutContext(seconds):
                    result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


@contextlib.contextmanager
def time_limit(seconds: float, on_end: Callable = None):
    """Sets a time limit."""
    if seconds == -1:
        yield
        return

    if ON_WINDOWS:
        with TimeoutContext(seconds, on_end):
            yield
    else:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, timeout_signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


# Original code from https://github.com/openai/human-eval/
class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


SUPPORTED_MODULES = {
    "safeguard": ("import safeguard", "safeguard.reliability_guard()"),
}


def get_module_and_call(module_name: str) -> Tuple[str, str, str]:
    """Gets the import code, call, and module source code for a module.

    Args:
        module_name (str): The name of the module to get.

    Returns:
        Tuple[str,str, str]: Import code, call, and module source code.
    """
    if module_name not in SUPPORTED_MODULES:
        raise ValueError(f"Module '{module_name}' is supported.")

    import_call, call = SUPPORTED_MODULES[module_name]
    module_source = inspect.getsource(getattr(utility_modules, module_name))
    return import_call, call, module_source


def get_results_from_generator(
    generator: Generator,
    total: int,
    target_returns_multiple: bool,
    garbage_collect_freq: int,
    log_freq: int,
):
    """Gets the results from a generator.

    Args:
        generator (Generator): The generator to get results from.
        total (int): The total number of items in the generator.
        target_returns_multiple (bool): If the target returns multiple items per iteration.
        disable_tqdm (bool): Whether to disable the progress bar.
        garbage_collect_freq (int): How often to perform garbage collection.
        log_freq (int): How often to log if not using tqdm.
        quiet (bool, optional): Whether to suppress logging. Defaults to False.

    Returns:
        List: The results from the generator.
    """
    results = []

    # Create a counter for completed since the size of results will not
    # always go up by one.
    start_time = time.time()
    num_completed = 0
    for r in generator:
        if target_returns_multiple:
            results.extend(r)
        else:
            results.append(r)
        num_completed += 1
        if num_completed % log_freq == 0:
            logger.debug(f"Finished {num_completed}/{total}")

        if num_completed % garbage_collect_freq == 0:
            gc.collect()
    elapsed = time.time() - start_time
    return elapsed, results


def run_in_parallel(
    target: Callable,
    args: List,
    num_workers: int,
    desc: Optional[str] = None,
    max_tasks_per_process: Optional[int] = None,
    disable_tqdm: bool = False,
    garbage_collect_freq: int = 500,
    chunk_size: int = 1,
    log_freq: int = 500,
    target_returns_multiple: bool = False,
    tqdm_kwargs: Optional[Dict] = None,
) -> List:
    """Runs a function in parallel.

    Args:
        target (Callable): The function to run.
        args (List): The arguments to pass to the function.
        num_workers (int): The number of workers to use.
        desc (str): The description to use for the progress bar.
        max_tasks_per_process (Optional[int], optional): Maximum number of tasks
            before starting a new process. Defaults to None.
        disable_tqdm (bool, optional): Disable the progress bar. Defaults to False.
        garbage_collect_freq (int, optional): How often to perform garbage
            collection. Defaults to 500.
        chunk_size (int, optional): The chunk size to use for imap. Defaults to 1.
        log_freq (int, optional): How often to log if not using tqdm. Defaults
            to 500.
        target_returns_multiple (bool, optional): If the target returns multiple
            so that `.extend` is used instead of `.append`. Defaults to False.
        tqdm_kwargs (Optional[Dict], optional): Additional keyword arguments to
            pass to tqdm. Defaults to None.

    Returns:
        List: The results of `target(a)` for each `a` in `args`.
    """
    logger.debug(
        "Starting run_in_parallel for %s.",
        desc or getattr(target, "__name__", "Target"),
    )

    logger.debug("Will use %d/%d CPUs", num_workers, mp.cpu_count())

    generator_creator = functools.partial(
        tqdm,
        total=len(args),
        desc=desc,
        disable=disable_tqdm,
        **(tqdm_kwargs or {}),
    )

    num_workers = min(num_workers, len(args))

    # Yes, this is not entirely parallel, but it makes debugging so much easier.
    if num_workers > 1:
        logger.debug(
            "Running in parallel with %d workers (%s,%s)",
            num_workers,
            f"{max_tasks_per_process=}",
            f"{chunk_size=}",
        )
        with mp.Pool(
            processes=num_workers, maxtasksperchild=max_tasks_per_process
        ) as pool:
            pbar_generator = generator_creator(
                pool.imap(target, args, chunksize=chunk_size),
            )
            elapsed, results = get_results_from_generator(
                generator=pbar_generator,
                total=len(args),
                target_returns_multiple=target_returns_multiple,
                garbage_collect_freq=garbage_collect_freq,
                log_freq=log_freq,
            )
            # Cleanup pool
            pool.close()
            pool.terminate()
    else:
        logger.debug("Running in serial as num_workers=1")
        pbar_generator = generator_creator(map(target, args))
        elapsed, results = get_results_from_generator(
            generator=pbar_generator,
            total=len(args),
            target_returns_multiple=target_returns_multiple,
            garbage_collect_freq=garbage_collect_freq,
            log_freq=log_freq,
        )

    pbar_generator.close()
    logger.debug(f"Finished {desc} in {elapsed:.2f} seconds")
    return results


def configure_logging(
    level: int = logging.DEBUG, format: str = None, datefmt: str = None
):
    root = logging.getLogger("code_execution")
    root.propagate = True
    root.setLevel(level)
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if datefmt is None:
        datefmt = "%Y-%m-%d %H:%M:%S"
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(format, datefmt=datefmt))


MEM_MAX_CODE = """__MAX_MEM = %%MEM_LIMIT%%
def _set_mem_limit():
    import resource
    import platform

    if not __MAX_MEM:
        return
    resource.setrlimit(
        resource.RLIMIT_AS, (__MAX_MEM, __MAX_MEM)
    )
    resource.setrlimit(
        resource.RLIMIT_DATA, (__MAX_MEM, __MAX_MEM)
    )
    if not platform.uname().system == "Darwin":
        resource.setrlimit(
            resource.RLIMIT_STACK, (__MAX_MEM, __MAX_MEM)
        )

_set_mem_limit()
"""


def get_mem_limit_code(mem_limit: Optional[str], trailing: str = "\n") -> str:
    """Gets the code to set the memory limit.

    Args:
        mem_limit (str): The memory limit value as a string. You can do
            something like "4 * 1024" or "1024". If None, will return an
            empty string.
        trailing: The trailing characters to add to the code.
    Returns:
        str: The code to set the memory limit.
    """
    if mem_limit is None:
        return ""

    out = MEM_MAX_CODE.replace("%%MEM_LIMIT%%", mem_limit)
    return out + trailing
