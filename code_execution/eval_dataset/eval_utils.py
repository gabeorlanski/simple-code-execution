import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import ujson

from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.data_structures import default_should_early_stop
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig
from code_execution.utils import get_mem_limit_code

logger = logging.getLogger(__name__)


def make_stdin_executable(
    files: Dict[str, str],
    inputs: List[List[str]],
    commands: List[str] | List[List[str]],
    first_command_timeout: float,
    early_stop_fn: Callable,
    ensure_all_run: bool = False,
    tracked_files: List[str] = None,
    command_timeout: float = 2.0,
    stdout_postprocess_fn: Optional[Callable[[str], str]] = None,
) -> Executable:
    """Makes an executable for an stdin program.

    Args:
        files: The files.
        inputs: The inputs.
        commands: The commands.
        early_stop_fn: The early stop function.
        ensure_all_run: Whether to ensure all commands are run.
        tracked_files: The tracked files.
        first_command_timeout: The timeout for the first command. If not set, the default timeout is used.
        command_timeout: The timeout for the commands.
        stdout_postprocess_fn: The stdout postprocess function.

    Returns:
        The executable.
    """
    if isinstance(commands[0], str):
        commands = [commands] * len(inputs)
    if len(commands) != len(inputs):
        raise ValueError("Number of commands must match number of inputs")

    commands = [
        Command(
            command=c,
            timeout=(
                max(first_command_timeout, command_timeout)
                if i == 0
                else command_timeout
            ),
            stdin=stdin,
        )
        for i, (stdin, c) in enumerate(zip(inputs, commands))
    ]

    return Executable(
        files=files,
        commands=commands,
        ensure_all_run=ensure_all_run,
        tracked_files=tracked_files or [],
        should_early_stop=early_stop_fn,
        stdout_postprocessor=stdout_postprocess_fn,
    )
