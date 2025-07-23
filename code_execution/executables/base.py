from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel
from pydantic import Field


class Command(BaseModel):
    """A command to be executed."""

    command: List[str]
    timeout: Optional[int] = None
    stdin: List[str] = Field(default_factory=list)


class BaseResult(BaseModel):
    """The base result class for all executables."""

    output: str
    return_code: int
    runtime: float
    timed_out: bool
    had_unexpected_error: bool = False

    @property
    def had_error(self) -> bool:
        """Whether the command had an error."""
        return (
            self.return_code != 0 or self.timed_out or self.had_unexpected_error
        )


class ExecutableResult(BaseModel):
    """The result of executing an executable."""

    results: List[BaseResult]
    elapsed: float
    tracked_files: Dict[str, str]


class BaseExecutable(BaseModel):
    """The Base Executable class that all executables should inherit from."""

    files: Dict[str, str]
    commands: List[Command]
    early_stopping: bool = False
    tracked_files: List[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        """Returns a string representation of the executable."""
        return f"{self.__class__.__name__}({len(self.files)} File(s), {len(self.commands)} Command(s))"


class RunnerRegistry:
    """The registry for all runners."""

    runners: Dict[str, Callable[[BaseExecutable], ExecutableResult]] = {}

    @classmethod
    def register(cls, name: str) -> None:
        """Registers a runner."""

        def wrapper(func: Callable[[BaseExecutable], ExecutableResult]) -> None:
            cls.runners[name] = func
            return func

        return wrapper

    @classmethod
    def get_runner(
        cls, name: str
    ) -> Callable[[BaseExecutable], ExecutableResult]:
        """Gets a runner."""
        return cls.runners[name]
