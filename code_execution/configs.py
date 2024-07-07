""" This file contains the configuration for the code execution module. """

from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Optional


@dataclass
class ExecutionConfig:
    """Config for execution.
    Args:
        num_workers: The number of workers to use.
        batch_size: The batch size to use for pre- and post- processing.
        max_tasks_per_process: The maximum number of tasks to run per process. If not none,
            the worker will be killed every `max_tasks_per_process` and a new one
            will be created.
        write_rate_limit: The rate limit for writing files. By default it is 768.
        chunk_size: The chunk size for parallel execution.
        batch_size: The batch size to use for all parallel operations.
        disable_tqdm: Whether to disable tqdm.
        execution_chunk_size: The chunk size for execution.
        default_timeout: The default timeout for execution.
        max_execute_at_once: The maximum number of predictions to execute at a single time.
        num_executors: The number of executor processes running.
        log_freq: How often to log progress.
        buffer_size: Chunk size to use for execution.
        display_write_progress: Display progress bars for writing and cleaning up.
        write_log_freq: Frequency for writing log messages.
    """

    num_workers: int
    max_tasks_per_process: Optional[int] = None
    write_rate_limit: int = 768
    chunk_size: int = 1
    batch_size: int = 1
    disable_tqdm: bool = False
    default_timeout: int = 10
    max_execute_at_once: int = -1
    num_executors: int = 4
    log_freq: int = 1000
    buffer_size: int = 100
    display_write_progress: bool = False
    write_log_freq: int = 100_000

    def __post_init__(self):
        if self.num_workers < 1 or self.num_workers >= cpu_count():
            self.num_workers = cpu_count() - 4

    @property
    def batched(self):
        """Whether to use batched processing."""
        return self.batch_size > 1
