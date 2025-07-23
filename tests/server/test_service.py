"""Unit tests for the execution service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from code_execution.executables.base import Command
from code_execution.executables.base import ExecutableResult
from code_execution.executables.subproc import SubprocessExecutable
from code_execution.executables.subproc import SubprocessResult
from code_execution.server.models import ExecuteRequest
from code_execution.server.service import ExecutionService


class TestDetectExecutableType:
    """Test cases for executable type detection."""

    def test_detect_executable_type_subprocess(self):
        """Test detection of subprocess executable type."""
        request = ExecuteRequest(
            files={"test.py": "print('hello')"},
            commands=[Command(command=["python", "test.py"])],
            executable_type="subprocess",
        )

        result = ExecutionService.detect_executable_type(request)

        assert result == "subprocess"

    def test_detect_executable_type_default(self):
        """Test default executable type when none specified."""
        request = ExecuteRequest(
            files={"test.py": "print('hello')"},
            commands=[Command(command=["python", "test.py"])],
        )

        result = ExecutionService.detect_executable_type(request)

        assert result == "subprocess"

    def test_detect_executable_type_unsupported(self):
        """Test fallback to default for unsupported type."""
        request = ExecuteRequest(
            files={"test.py": "print('hello')"},
            commands=[Command(command=["python", "test.py"])],
            executable_type="unsupported_type",
        )

        result = ExecutionService.detect_executable_type(request)

        assert result == "subprocess"


class TestValidateRequest:
    """Test cases for request validation."""

    def test_validate_request_valid(self):
        """Test validation of valid request."""
        request = ExecuteRequest(
            files={"test.py": "print('hello')"},
            commands=[Command(command=["python", "test.py"])],
        )

        # Should not raise any exception
        ExecutionService.validate_request(request)

    def test_validate_request_no_commands(self):
        """Test validation fails with no commands."""
        request = ExecuteRequest(
            files={"test.py": "print('hello')"}, commands=[]
        )

        with pytest.raises(
            ValueError, match="At least one command is required"
        ):
            ExecutionService.validate_request(request)

    def test_validate_request_no_files_with_file_commands(self):
        """Test validation warning when commands reference files but no files provided."""
        request = ExecuteRequest(
            files={}, commands=[Command(command=["python", "script.py"])]
        )

        # Should not raise exception, but may log warning
        ExecutionService.validate_request(request)

    def test_validate_request_valid_no_file_references(self):
        """Test validation with commands that don't reference files."""
        request = ExecuteRequest(
            files={}, commands=[Command(command=["echo", "hello"])]
        )

        # Should not raise any exception
        ExecutionService.validate_request(request)


class TestExecuteRequest:
    """Test cases for request execution."""

    @pytest.mark.asyncio
    @patch("code_execution.server.service.RunnerRegistry.get_runner")
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_success(
        self, mock_get_limiter, mock_get_runner
    ):
        """Test successful request execution."""
        # Setup mocks
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="Hello World!\n",
                    return_code=0,
                    runtime=0.1,
                    timed_out=False,
                    stderr="",
                )
            ],
            elapsed=0.12,
            tracked_files={},
        )
        mock_runner = AsyncMock(return_value=mock_result)
        mock_get_runner.return_value = mock_runner

        # Mock concurrency limiter
        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock(return_value="exec_123")
        mock_limiter.release = AsyncMock()
        mock_limiter.get_queue_info.return_value = {"current_executions": 1}
        mock_get_limiter.return_value = mock_limiter

        # Create request
        request = ExecuteRequest(
            files={"hello.py": "print('Hello World!')"},
            commands=[Command(command=["python", "hello.py"], timeout=30)],
            executable_type="subprocess",
        )

        # Execute request
        result, metadata = await ExecutionService.execute_request(request)

        # Assert results
        assert result == mock_result
        assert metadata["execution_id"] == "exec_123"
        mock_get_runner.assert_called_once_with("subprocess")
        mock_runner.assert_called_once()
        mock_limiter.acquire.assert_called_once()
        mock_limiter.release.assert_called_once_with("exec_123")

        # Verify the executable passed to runner
        called_executable = mock_runner.call_args[0][0]
        assert isinstance(called_executable, SubprocessExecutable)
        assert called_executable.files == request.files
        assert called_executable.commands == request.commands

    @pytest.mark.asyncio
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_validation_error(self, mock_get_limiter):
        """Test request execution with validation error."""
        # Create invalid request
        request = ExecuteRequest(
            files={"test.py": "print('test')"}, commands=[]  # No commands
        )

        # Execute request should raise ValueError
        with pytest.raises(
            ValueError, match="At least one command is required"
        ):
            await ExecutionService.execute_request(request)

        # Limiter should not be called for invalid requests
        mock_get_limiter.assert_not_called()

    @pytest.mark.asyncio
    @patch("code_execution.server.service.RunnerRegistry.get_runner")
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_no_runner(
        self, mock_get_limiter, mock_get_runner
    ):
        """Test request execution when no runner found."""
        # Setup mocks
        mock_get_runner.side_effect = KeyError("subprocess")

        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock(return_value="exec_123")
        mock_limiter.release = AsyncMock()
        mock_limiter.get_queue_info.return_value = {"current_executions": 1}
        mock_get_limiter.return_value = mock_limiter

        # Create request
        request = ExecuteRequest(
            files={"test.py": "print('test')"},
            commands=[Command(command=["python", "test.py"])],
        )

        # Execute request should raise RuntimeError
        with pytest.raises(
            RuntimeError, match="No runner found for executable type"
        ):
            await ExecutionService.execute_request(request)

        # Should still release the acquired slot
        mock_limiter.release.assert_called_once_with("exec_123")

    @pytest.mark.asyncio
    @patch("code_execution.server.service.RunnerRegistry.get_runner")
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_runner_failure(
        self, mock_get_limiter, mock_get_runner
    ):
        """Test request execution when runner fails."""
        # Setup mocks
        mock_runner = AsyncMock(side_effect=Exception("Runner failed"))
        mock_get_runner.return_value = mock_runner

        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock(return_value="exec_123")
        mock_limiter.release = AsyncMock()
        mock_limiter.get_queue_info.return_value = {"current_executions": 1}
        mock_get_limiter.return_value = mock_limiter

        # Create request
        request = ExecuteRequest(
            files={"test.py": "print('test')"},
            commands=[Command(command=["python", "test.py"])],
        )

        # Execute request should raise RuntimeError
        with pytest.raises(RuntimeError, match="Execution failed"):
            await ExecutionService.execute_request(request)

        # Should still release the acquired slot
        mock_limiter.release.assert_called_once_with("exec_123")

    @pytest.mark.asyncio
    @patch("code_execution.server.service.RunnerRegistry.get_runner")
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_with_early_stopping(
        self, mock_get_limiter, mock_get_runner
    ):
        """Test request execution with early stopping enabled."""
        # Setup mocks
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="",
                    return_code=1,
                    runtime=0.05,
                    timed_out=False,
                    stderr="Error occurred",
                )
            ],
            elapsed=0.06,
            tracked_files={},
        )
        mock_runner = AsyncMock(return_value=mock_result)
        mock_get_runner.return_value = mock_runner

        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock(return_value="exec_123")
        mock_limiter.release = AsyncMock()
        mock_limiter.get_queue_info.return_value = {"current_executions": 1}
        mock_get_limiter.return_value = mock_limiter

        # Create request with early stopping
        request = ExecuteRequest(
            files={"test.py": "raise ValueError('test error')"},
            commands=[
                Command(command=["python", "test.py"]),
                Command(command=["echo", "should not run"]),
            ],
            early_stopping=True,
        )

        # Execute request
        result, metadata = await ExecutionService.execute_request(request)

        # Assert results
        assert result == mock_result

        # Verify executable was configured with early stopping
        called_executable = mock_runner.call_args[0][0]
        assert called_executable.early_stopping is True

    @pytest.mark.asyncio
    @patch("code_execution.server.service.RunnerRegistry.get_runner")
    @patch("code_execution.server.service.get_concurrency_limiter")
    async def test_execute_request_with_tracked_files(
        self, mock_get_limiter, mock_get_runner
    ):
        """Test request execution with tracked files."""
        # Setup mocks
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="File created\n",
                    return_code=0,
                    runtime=0.02,
                    timed_out=False,
                    stderr="",
                )
            ],
            elapsed=0.03,
            tracked_files={"output.txt": "Generated content"},
        )
        mock_runner = AsyncMock(return_value=mock_result)
        mock_get_runner.return_value = mock_runner

        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock(return_value="exec_123")
        mock_limiter.release = AsyncMock()
        mock_limiter.get_queue_info.return_value = {"current_executions": 1}
        mock_get_limiter.return_value = mock_limiter

        # Create request with tracked files
        request = ExecuteRequest(
            files={
                "script.py": "with open('output.txt', 'w') as f: f.write('Generated content')"
            },
            commands=[Command(command=["python", "script.py"])],
            tracked_files=["output.txt"],
        )

        # Execute request
        result, metadata = await ExecutionService.execute_request(request)

        # Assert results
        assert result == mock_result
        assert "output.txt" in result.tracked_files

        # Verify executable was configured with tracked files
        called_executable = mock_runner.call_args[0][0]
        assert called_executable.tracked_files == ["output.txt"]
