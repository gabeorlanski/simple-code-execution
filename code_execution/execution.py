""" Module for executing code. """

import concurrent.futures
import dataclasses
import functools
import logging
import multiprocessing as mp
import pathlib
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import psutil
from tqdm import tqdm

from .configs import ExecutionConfig
from .utils import get_results_from_generator

LOGGING_IS_CONFIGURED = logging.getLogger().hasHandlers()


logger = logging.getLogger(__name__)


def seconds_to_human(seconds):
    """Converts seconds to a human readable format."""
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Dataclass for the result of executing a command.

    Args:
        return_code: The return code.
        runtime: The runtime.
        stdout: The stdout.
        stderr: The stderr.
        timed_out: Whether the command timed out.
        had_unexpected_error: Whether the command had an unexpected error.
    """

    return_code: int
    runtime: float
    stdout: str
    stderr: str
    timed_out: bool
    had_unexpected_error: bool = False


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Dataclass for the result of executing a list of commands.

    Args:
        command_results: The results of the commands.
        elapsed: The elapsed time.
        cwd: The current working directory.
        tracked_files: The tracked files.
    """

    command_results: List[CommandResult]
    elapsed: float
    cwd: str
    tracked_files: Dict[str, str]

    @property
    def timed_out(self) -> bool:
        """Whether the last command timed out."""
        if not self.command_results:
            return False
        return self.command_results[-1].timed_out

    @property
    def had_error(self) -> bool:
        """Whether the last command had an error."""
        if not self.command_results:
            return True
        return self.command_results[-1].return_code != 0

    @property
    def last_cmd(self) -> CommandResult:
        """The last command result."""
        if not self.command_results:
            return None
        return self.command_results[-1]

    def to_dict(self) -> Dict:
        """Converts the result to a dictionary."""
        return {
            "command_results": [
                dataclasses.asdict(r) for r in self.command_results
            ],
            "cwd": self.cwd,
            "tracked_files": self.tracked_files,
            "elapsed": self.elapsed,
        }

    @classmethod
    def invalid_result(
        cls,
        num_commands: int = 1,
        runtime: float = 10.0,
        return_code: int = 1,
        stdout: str = "SyntaxError",
        stderr: str = "Invalid",
        elapsed: float = 10.0,
    ) -> "ExecutionResult":
        """Creates a dummy ExecutionResult that represents an invalid result.
        Useful for when your preprocessor finds a program you want to skip
        execution for."""
        return cls(
            command_results=[
                CommandResult(
                    return_code=return_code,
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=False,
                )
                for _ in range(num_commands)
            ],
            elapsed=elapsed,
            cwd=None,
            tracked_files={},
        )


def _execute(
    command_to_run: List[str], working_dir: pathlib.Path, timeout: int
) -> Dict:
    """Executes a single command."""
    timed_out = False
    return_code = -1
    runtime = timeout
    stderr = None
    stdout = None
    had_unexpected_error = False
    start_time = time.time()
    execution_process = subprocess.Popen(
        command_to_run,
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        try:
            outputs = execution_process.communicate(timeout=timeout)
            t1 = time.time()
            stdout = outputs[0].decode("utf-8")
            stderr = outputs[1].decode("utf-8")
            runtime = t1 - start_time
            return_code = execution_process.returncode

        except subprocess.TimeoutExpired:
            stdout = stderr = ""
            runtime = timeout
            return_code = 0
            timed_out = True
        execution_process.kill()

    # pylint: disable=broad-except
    except Exception as e:
        stderr = str(e)
        stdout = ""
        return_code = -1
        runtime = -1
        timed_out = False
        had_unexpected_error = True
        execution_process.kill()
    return dict(
        return_code=return_code,
        runtime=runtime,
        stderr=stderr,
        stdout=stdout,
        timed_out=timed_out,
        had_unexpected_error=had_unexpected_error,
    )


def safe_execute(
    command_to_run: List[str],
    working_dir: pathlib.Path,
    timeout: int = 10,
    num_times: int = 1,
) -> CommandResult:
    """Executes a list of commands safely.
    Args:
      command_to_run: The command to run.
      working_dir: The working directory to run them in.
      timeout Timeout.
      num_times: Number of times to execute the command. Useful for getting
        runtime and memory means.
    Returns:
      The result of executing the command.
    """
    times = []
    had_error = False
    for _ in range(num_times):
        res = _execute(command_to_run, working_dir, timeout)
        times.append(res["runtime"])
        if res["return_code"] != 0:
            had_error = True
        if res["timed_out"]:
            had_error = True
        if res["had_unexpected_error"]:
            had_error = True

        if had_error:
            break

    if num_times == 1 or had_error:
        res["runtime"] = times[0]
    else:
        res["runtime"] = float(np.mean(times))

    return CommandResult(**res)


def serial_execute_code(sample: Dict) -> ExecutionResult:
    """Execute a file of code.
    Args:
        sample: The sample to run.
    Returns:
        The execution result.
    """
    file_path = sample["cwd"]
    working_dir_for_execution = (
        file_path.parent if file_path.is_file() else file_path
    )

    working_dir_for_execution = working_dir_for_execution.resolve().absolute()
    results = []
    t0 = time.time()
    for command in sample["commands"]:
        res = safe_execute(
            command["command"],
            working_dir=working_dir_for_execution,
            timeout=command["timeout"],
            num_times=command.get("num_times", 1),
        )
        results.append(res)
        if res.timed_out:
            break
        if res.return_code != 0:
            break

    file_contents = {}
    for fn in sample["tracked_files"]:
        fp = file_path.joinpath(fn)
        if fp.exists():
            file_contents[fn] = fp.read_text(encoding="utf-8")
        else:
            file_contents[fn] = None
    elapsed = time.time() - t0
    return ExecutionResult(
        command_results=results,
        elapsed=elapsed,
        cwd=str(working_dir_for_execution),
        tracked_files=file_contents,
    )


def execute_single(execution_dict: Dict) -> Tuple[Tuple, ExecutionResult]:
    """Executes a single program."""
    key = execution_dict["key"]
    executable = execution_dict["executable"]
    return key, serial_execute_code(executable)


def batched_execute_code(to_run: List[Dict]) -> List[Dict]:
    """Executes a batch of commands."""
    results = [None] * len(to_run)
    for i, command in enumerate(to_run):
        results[i] = execute_single(command)
    return results


def sizeof_fmt(num, suffix="B"):
    """Human readable file size."""
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def threaded_execution(
    to_run, execution_fn, max_threads, is_batched: bool = False
):
    """Executes a list of commands in parallel."""
    num_threads = min(len(to_run), max_threads)
    out = []
    if max_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            for result in executor.map(execution_fn, to_run):
                if is_batched:
                    out.extend(result)
                else:
                    out.append(result)

            executor.shutdown()
    else:
        for command in to_run:
            if is_batched:
                out.extend(execution_fn(command))
            else:
                out.append(execution_fn(command))
    return out


def _parallel_execute_code(
    to_run: List,
    max_processes: int,
    num_executors: int,
    log_freq: int,
    is_batched: bool = False,
    execute_batch_size: int = 100,
) -> List[ExecutionResult]:
    """Executes a list of commands in parallel.

    Args:
        to_run: The list of commands to run.
        max_processes: The maximum number of processes to run.
        num_executors: The number of executors to run.
        log_freq: The frequency to log progress.
        is_batched: Whether the commands are batched.
        execute_batch_size: The size of execution batches.

    Returns:
        The list of results.
    """
    logger.info(
        "Starting parallel execution (max_processes=%d num_executors=%d)",
        max_processes,
        num_executors,
    )
    max_threads = max(max_processes // num_executors, 1)
    logger.debug("max_threads=%d", max_threads)
    manager_process = psutil.Process()
    chunk_size = min(execute_batch_size, len(to_run) // max_threads)
    logger.debug("chunk_size=%d", chunk_size)
    # initialize cpu percent
    psutil.getloadavg()
    chunks = []
    if is_batched:
        total_commands = sum(map(len, to_run))
    else:
        total_commands = len(to_run)
    for i in range(0, len(to_run), chunk_size):
        chunks.append(to_run[i : i + chunk_size])
    logger.info(
        f"Executing {total_commands:,} command(s) in {len(chunks):,} chunk(s)"
    )
    results = []
    start_time = time.time()
    interval_start = start_time
    interval_received = 0
    executor_fn = batched_execute_code if is_batched else execute_single
    threaded_fn = functools.partial(
        threaded_execution,
        execution_fn=executor_fn,
        max_threads=max_threads,
        is_batched=is_batched,
    )

    progress_writer = (
        logger.info
        if LOGGING_IS_CONFIGURED
        else lambda m: print(f"{datetime.now().isoformat(' ','seconds')} {m}")
    )

    last_log = 0
    with mp.Pool(processes=num_executors) as pool:
        for result in pool.imap_unordered(threaded_fn, chunks):
            results.extend(result)
            if len(results) - last_log >= log_freq:
                last_log = len(results)
                t1 = time.time()
                interval_elapsed = t1 - interval_start
                elapsed = t1 - start_time
                interval_completed = len(results) - interval_received
                prog = len(results) / total_commands
                rate = interval_completed / interval_elapsed
                eta = seconds_to_human((total_commands - len(results)) / rate)
                interval_received = len(results)
                rate_str = f"{rate:0.2f} P/S"
                prog_str = f"{prog:0.2%}"
                progress_writer(
                    f"{len(results):>9,}/{total_commands:<9,}({prog_str:>6}) @ {rate_str:<12}"
                    f" in {seconds_to_human(elapsed)} ETA: {eta}"
                )
                one_min_cpu, _, fifteen_min_cpu = [
                    x / psutil.cpu_count() for x in psutil.getloadavg()
                ]

                logger.debug(
                    f"Memory={sizeof_fmt(manager_process.memory_info().rss)} "
                    f"CPU: 1Min={one_min_cpu:0.2%} 15Min={fifteen_min_cpu:0.2%}"
                )
                interval_start = time.time()

        pool.close()
        pool.terminate()

    progress_writer(
        f"Finished executing {len(results):,} in {seconds_to_human(time.time() - start_time)}"
    )

    if len(results) != total_commands:
        raise ValueError(
            f"Expected {total_commands:,} results, got {len(results):,}"
        )
    return results


def execute_commands(
    predictions,
    config: ExecutionConfig,
) -> List[ExecutionResult]:
    """Executes a list of commands."""
    if not LOGGING_IS_CONFIGURED:
        print(f"Executing {len(predictions):,} predictions")
    else:
        logger.debug("Executing %d predictions", len(predictions))
    if config.batched:
        logger.debug(
            "Running in batched mode with batch_size=%d", config.batch_size
        )
        executor_fn = batched_execute_code
        to_run = []
        for i in range(0, len(predictions), config.batch_size):
            to_run.append(predictions[i : i + config.batch_size])
    else:
        logger.debug("Running in non-batched mode")
        executor_fn = execute_single
        to_run = predictions
    num_workers = min(len(to_run), config.num_workers)

    # Yes, this is not entirely parallel, but it makes debugging so much easier.
    if num_workers > 1:
        results = _parallel_execute_code(
            to_run=to_run,
            max_processes=num_workers,
            num_executors=config.num_executors,
            is_batched=config.batched,
            log_freq=config.log_freq,
            execute_batch_size=config.buffer_size,
        )
    else:
        logger.debug("Running in serial as num_workers=1")
        pbar_generator = tqdm(
            map(executor_fn, to_run),
            total=len(to_run),
            desc="Executing Predictions",
            disable=config.disable_tqdm,
        )
        results = get_results_from_generator(
            generator=pbar_generator,
            total=len(to_run),
            target_returns_multiple=config.batched,
            garbage_collect_freq=500,
            log_freq=500,
        )

        pbar_generator.close()
    return results
