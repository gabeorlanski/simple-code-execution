"""API models for the FastAPI server."""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from pydantic import Field

from code_execution.executables.base import BaseResult
from code_execution.executables.base import Command
from code_execution.executables.base import ExecutableResult


class ExecuteRequest(BaseModel):
    """Request model for the /execute endpoint."""

    files: Dict[str, str] = Field(
        description="Dictionary mapping filenames to their content"
    )
    commands: List[Command] = Field(
        description="List of commands to execute sequentially"
    )
    executable_type: Optional[str] = Field(
        default="subprocess",
        description="Type of executable to use (defaults to 'subprocess')"
    )
    early_stopping: bool = Field(
        default=False,
        description="Whether to stop execution on first command failure"
    )
    tracked_files: List[str] = Field(
        default_factory=list,
        description="List of files to track after execution"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Execution priority (1=highest, 10=lowest)"
    )


class ExecuteResponse(BaseModel):
    """Response model for successful execution."""

    success: bool = Field(default=True, description="Whether execution succeeded")
    result: ExecutableResult = Field(description="Execution result details")
    queue_position: Optional[int] = Field(
        default=None, description="Position in execution queue (if queued)"
    )
    estimated_wait_time: Optional[float] = Field(
        default=None, description="Estimated wait time in seconds (if queued)"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    success: bool = Field(default=False, description="Whether execution succeeded")
    error: str = Field(description="Error message")
    details: Optional[str] = Field(
        default=None, description="Additional error details"
    )


# Type alias for API responses
ApiResponse = Union[ExecuteResponse, ErrorResponse]