import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing as mp
import signal
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple


from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_normal
from code_execution import utility_modules

LOGGING_IS_CONFIGURED = logging.getLogger().hasHandlers()


logger = logging.getLogger(__name__)


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


TQDM_CLASS = tqdm_notebook if in_notebook() else tqdm_normal


def wrap_pbar(iterable, **kwargs):
    return TQDM_CLASS(iterable, **kwargs)


def _batched_wrapper(batch, processor, proc_returns_list):
    """Wrapper for batched processing."""
    out = []
    for example in batch:
        idx = example["idx"]
        result = processor(*example["args"])
        if proc_returns_list:
            out.extend([((idx, i), r) for i, r in enumerate(result)])
        else:
            out.append(((idx, 0), result))
    return out


def _normal_wrapper(arg_dict, processor, proc_returns_list):
    """Wrapper for normal processing."""
    result = processor(*arg_dict["args"])
    if proc_returns_list:
        return [((arg_dict["idx"], i), r) for i, r in enumerate(result)]
    return ((arg_dict["idx"], 0), result)


def wrap_processor(
    processor_fn: Callable,
    batch_size: int,
    returns_list: bool,
) -> Callable:
    """Wraps a processor function to handle batching."""
    if batch_size > 1:
        wrapper = _batched_wrapper
    else:
        wrapper = _normal_wrapper
    return functools.partial(
        wrapper, processor=processor_fn, proc_returns_list=returns_list
    )


def get_pred_dir(idx: int, parent: Path):
    """Gets the prediction directory for a prediction."""
    return parent.joinpath(f"pred{idx}")


class ContextTimeLimitException(Exception):
    """Timeout error for running commands."""


@contextlib.contextmanager
def time_limit(seconds: float):
    """Sets a time limit."""

    def signal_handler(signum, frame):
        raise ContextTimeLimitException("Timed out!")

    if seconds != -1:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        if seconds != -1:
            signal.setitimer(signal.ITIMER_REAL, 0)


@dataclass(frozen=True)
class Executable:
    """Dataclass to represent the commands and setup needed to execute a prediction."""

    files: Dict[str, str]
    commands: List[Dict]
    tracked_files: List[str] = field(default_factory=list)


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
    disable_tqdm: bool,
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

    Returns:
        List: The results from the generator.
    """
    results = []

    # Create a counter for completed since the size of results will not
    # always go up by one.
    write_fn = logger.info if LOGGING_IS_CONFIGURED else logger.debug
    num_completed = 0
    for r in generator:
        if target_returns_multiple:
            results.extend(r)
        else:
            results.append(r)
        num_completed += 1
        if disable_tqdm and num_completed % log_freq == 0:
            write_fn(f"Finished {num_completed}/{total}")

        if num_completed % garbage_collect_freq == 0:
            gc.collect()
    return results


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

    Returns:
        List: The results of `target(a)` for each `a` in `args`.
    """
    logger.debug(
        "Starting run_in_parallel for %s",
        desc or getattr(target, "__name__", "Target"),
    )

    generator_creator = functools.partial(
        TQDM_CLASS, total=len(args), desc=desc, disable=disable_tqdm
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
            results = get_results_from_generator(
                generator=pbar_generator,
                total=len(args),
                target_returns_multiple=target_returns_multiple,
                disable_tqdm=disable_tqdm,
                garbage_collect_freq=garbage_collect_freq,
                log_freq=log_freq,
            )
            # Cleanup pool
            pool.close()
            pool.terminate()
    else:
        logger.debug("Running in serial as num_workers=1")
        pbar_generator = generator_creator(map(target, args))
        results = get_results_from_generator(
            generator=pbar_generator,
            total=len(args),
            target_returns_multiple=target_returns_multiple,
            disable_tqdm=disable_tqdm,
            garbage_collect_freq=garbage_collect_freq,
            log_freq=log_freq,
        )

    pbar_generator.close()
    return results
