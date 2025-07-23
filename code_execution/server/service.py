"""Execution service for the FastAPI server."""

from typing import Dict, List, Tuple

import structlog

from code_execution.executables.base import ExecutableResult
from code_execution.executables.base import RunnerRegistry
from code_execution.executables.subproc import SubprocessExecutable
from code_execution.server.concurrency import get_concurrency_limiter
from code_execution.server.models import ExecuteRequest

logger = structlog.get_logger()

# Mapping of executable types to their corresponding classes
EXECUTABLE_TYPES = {
    "subprocess": SubprocessExecutable,
}

# Default executable type
DEFAULT_EXECUTABLE_TYPE = "subprocess"


class ExecutionService:
    """Service for handling code execution requests."""

    @staticmethod
    def detect_executable_type(request: ExecuteRequest) -> str:
        """
        Detect the appropriate executable type for the request.

        Args:
            request: The execution request

        Returns:
            The detected executable type
        """
        # For now, we only support subprocess execution
        # In the future, this could analyze file extensions, commands, etc.
        requested_type = request.executable_type or DEFAULT_EXECUTABLE_TYPE

        if requested_type not in EXECUTABLE_TYPES:
            logger.warning(
                "Unsupported executable type requested, falling back to default",
                requested_type=requested_type,
                default_type=DEFAULT_EXECUTABLE_TYPE,
            )
            return DEFAULT_EXECUTABLE_TYPE

        return requested_type

    @staticmethod
    def validate_request(request: ExecuteRequest) -> None:
        """
        Validate the execution request.

        Args:
            request: The execution request to validate

        Raises:
            ValueError: If the request is invalid
        """
        if not request.commands:
            raise ValueError("At least one command is required")

        if not request.files and any(
            any(
                arg.endswith(
                    (".py", ".txt", ".sh", ".js", ".java", ".cpp", ".c")
                )
                for arg in cmd.command
            )
            for cmd in request.commands
        ):
            logger.warning(
                "Commands reference files but no files provided",
                commands=[cmd.command for cmd in request.commands],
            )

    @staticmethod
    async def execute_request(
        request: ExecuteRequest,
    ) -> Tuple[ExecutableResult, Dict[str, any]]:
        """
        Execute a code execution request with concurrency control.

        Args:
            request: The execution request

        Returns:
            Tuple of (execution result, execution metadata)

        Raises:
            ValueError: If the request is invalid
            RuntimeError: If execution fails
        """
        logger.info(
            "Processing execution request",
            files=list(request.files.keys()),
            commands=[cmd.command for cmd in request.commands],
        )

        # Validate the request
        ExecutionService.validate_request(request)

        # Get concurrency limiter and acquire slot
        limiter = get_concurrency_limiter()

        # Check queue info before acquiring
        queue_info = limiter.get_queue_info()
        logger.debug("Queue status before acquisition", **queue_info)

        execution_id = None
        try:
            # Acquire execution slot (may queue if at capacity)
            execution_id = await limiter.acquire()

            logger.info("Acquired execution slot", execution_id=execution_id)

            # Detect executable type
            executable_type = ExecutionService.detect_executable_type(request)
            logger.debug(
                "Using executable type",
                execution_id=execution_id,
                executable_type=executable_type,
            )

            # Create the appropriate executable instance
            executable_class = EXECUTABLE_TYPES[executable_type]
            executable = executable_class(
                files=request.files,
                commands=request.commands,
                early_stopping=request.early_stopping,
                tracked_files=request.tracked_files,
            )

            # Get the runner and execute
            try:
                runner = RunnerRegistry.get_runner(executable_type)
                result = await runner(executable)

                logger.info(
                    "Execution completed successfully",
                    execution_id=execution_id,
                    elapsed=result.elapsed,
                    num_results=len(result.results),
                    tracked_files=list(result.tracked_files.keys()),
                )

                # Prepare execution metadata
                metadata = {
                    "execution_id": execution_id,
                    "queue_info": limiter.get_queue_info(),
                }

                return result, metadata

            except KeyError as e:
                error_msg = (
                    f"No runner found for executable type: {executable_type}"
                )
                logger.error(
                    error_msg,
                    execution_id=execution_id,
                    executable_type=executable_type,
                )
                raise RuntimeError(error_msg) from e

            except Exception as e:
                error_msg = f"Execution failed: {str(e)}"
                logger.error(
                    error_msg,
                    execution_id=execution_id,
                    error=str(e),
                    executable_type=executable_type,
                )
                raise RuntimeError(error_msg) from e

        finally:
            # Always release the slot
            if execution_id:
                await limiter.release(execution_id)
