"""Data structures for code execution."""

import dataclasses
import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class Command:
    """Dataclass for a command to execute.

    Args:
        command: The command to execute.
        timeout: The timeout for the command. If not set, the default timeout is used.
        num_times: Number of times to execute the command.
        stdin: The stdin for the command.
        ignore_error: Whether to ignore errors and continue execution.
    """

    command: List[str]
    timeout: Optional[float] = None
    num_times: int = 1
    stdin: List[str] = dataclasses.field(default_factory=list)
    ignore_error: bool = False


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

    @property
    def had_error(self) -> bool:
        """Whether the last command had an error."""
        return self.return_code != 0 or self.had_unexpected_error

    def __repr__(self):
        if self.timed_out:
            return f"CommandResult(return_code={self.return_code}, runtime={self.runtime}, timed_out={self.timed_out})"

        if self.stdout:
            use_str = self.stdout
        else:
            use_str = self.stderr

        return f"CommandResult(return_code={self.return_code}, runtime={self.runtime}, output={use_str[:50]}...)"


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Dataclass for the result of executing a list of commands.

    Args:
        key: The key for the result.
        command_results: The results of the commands.
        elapsed: The elapsed time.
        cwd: The current working directory.
        tracked_files: The tracked files.
        expected_num_commands: The expected number of commands ran.
    """

    key: str
    command_results: List[CommandResult]
    elapsed: float
    cwd: str
    tracked_files: Dict[str, str]
    expected_num_commands: int
    writing_time: float = -1
    cleanup_time: float = -1
    preprocess_time: float = -1

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

    def all_had_return_code(self, return_code: int) -> bool:
        """Whether all commands had the same return code."""
        return all(r.return_code == return_code for r in self.command_results)

    def to_dict(self) -> Dict:
        """Converts the result to a dictionary."""

        return dataclasses.asdict(self)

    @classmethod
    def invalid_result(
        cls,
        key: str,
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
            key=key,
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
            expected_num_commands=num_commands,
        )

    def __getitem__(self, idx: int) -> CommandResult:
        return self.command_results[idx]

    def __len__(self) -> int:
        return len(self.command_results)


def default_should_early_stop(*_, **_k) -> bool:
    _ = _k
    return False


@dataclasses.dataclass(frozen=True)
class Executable:
    """Dataclass to represent the commands and setup needed to execute a prediction.

    Args:
        files: The files to write.
        commands: The commands to run.
        tracked_files: The files to get contents of after execution.
        ensure_all_run: Whether to ignore errors and continue. This will override
            individual command settings. Default is False.
        should_early_stop: A function that takes the index of the command and the result, returning a bool if the execution should stop early. THIS MUST BE PICKLEABLE
    """

    files: Dict[str, str]
    commands: List[Command]
    tracked_files: List[str] = dataclasses.field(default_factory=list)
    ensure_all_run: bool = False
    should_early_stop: Callable[[int, CommandResult], bool] = (
        default_should_early_stop
    )
    stdout_postprocessor: Optional[Callable[[str], str]] = None

    def __post_init__(self):
        if not callable(self.should_early_stop):
            raise ValueError("should_early_stop must be callable")


@dataclasses.dataclass(frozen=True)
class CommandsToRun:
    """Dataclass to represent the information needed to run a command.

    The main reason to have this class is to avoid the need to pass around the
    raw files to every function.

    Args:
        cwd: The current working directory.
        commands: The commands to run.
        tracked_files: The files to get contents of after execution.
        ensure_all_run: Whether to ignore errors and continue. This will override
            individual command settings. Default is False.
    """

    cwd: Path
    commands: List[Command]
    tracked_files: List[str] = dataclasses.field(default_factory=list)
    ensure_all_run: bool = False
    should_early_stop: Callable[[int, CommandResult], bool] = (
        default_should_early_stop
    )
    stdout_postprocessor: Optional[Callable[[str], str]] = None


@dataclasses.dataclass
class OverallExecutionResults:
    results: List[Dict]
    net_time: float
    pure_exec_time: float
    execution_time: float
    writing_time: float
    postprocessing_time: float
    preprocessing_time: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.isoformat(
                datetime.datetime.now()
            )

    @property
    def timing_dict(self) -> Dict:
        return {
            "net_time": self.net_time,
            "pure_exec_time": self.pure_exec_time,
            "execution_time": self.execution_time,
            "writing_time": self.writing_time,
            "postprocessing_time": self.postprocessing_time,
            "preprocessing_time": self.preprocessing_time,
            "timestamp": self.timestamp,
        }
