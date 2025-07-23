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


MEM_MAX_CODE = """__MAX_MEM = %%MEM_LIMIT%%
def _set_mem_limit():
    import platform
    import resource

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
