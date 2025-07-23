"""Unit tests for the subprocess executable module."""

import asyncio
from pathlib import Path
import subprocess
import tempfile
import time
from typing import List
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from code_execution.executables.base import Command
from code_execution.executables.subproc import SubprocessExecutable
from code_execution.executables.subproc import SubprocessExecutableResult
from code_execution.executables.subproc import SubprocessResult
from code_execution.executables.subproc import _execute
from code_execution.executables.subproc import execute_subprocess_async


class TestSubprocessResult:
    """Test cases for SubprocessResult class."""

    def test_subprocess_result_initialization(self):
        """Test that SubprocessResult initializes correctly."""
        result = SubprocessResult(
            output="test output",
            return_code=0,
            runtime=1.5,
            timed_out=False,
            stderr="test stderr",
        )

        assert result.output == "test output"
        assert result.return_code == 0
        assert result.runtime == 1.5
        assert result.timed_out is False
        assert result.stderr == "test stderr"

    def test_subprocess_result_had_error_false(self):
        """Test had_error property returns False for successful execution."""
        result = SubprocessResult(
            output="success",
            return_code=0,
            runtime=1.0,
            timed_out=False,
            stderr="",
        )

        assert result.had_error is False

    def test_subprocess_result_had_error_true_return_code(self):
        """Test had_error property returns True for non-zero return code."""
        result = SubprocessResult(
            output="error",
            return_code=1,
            runtime=1.0,
            timed_out=False,
            stderr="error message",
        )

        assert result.had_error is True

    def test_subprocess_result_had_error_true_timeout(self):
        """Test had_error property returns True for timeout."""
        result = SubprocessResult(
            output="", return_code=0, runtime=5.0, timed_out=True, stderr=""
        )

        assert result.had_error is True


class TestSubprocessExecutableResult:
    """Test cases for SubprocessExecutableResult class."""

    def test_executable_type_is_subprocess(self):
        """Test that executable_type is set to 'subprocess'."""
        result = SubprocessExecutableResult(
            results=[], elapsed=1.0, tracked_files={}
        )

        assert result.executable_type == "subprocess"


class TestExecuteFunction:
    """Test cases for the _execute function."""

    @patch("subprocess.Popen")
    @patch("time.time")
    def test_execute_successful_command(self, mock_time, mock_popen):
        """Test successful command execution."""
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Execute
        result = _execute(
            command_to_run=["echo", "hello"],
            working_dir=Path("/tmp"),
            timeout=10,
        )

        # Assert
        assert result.output == "output"
        assert result.return_code == 0
        assert result.runtime == 1.5
        assert result.timed_out is False
        assert result.stderr == ""
        assert result.had_unexpected_error is False

    @patch("subprocess.Popen")
    @patch("time.time")
    def test_execute_command_with_stderr(self, mock_time, mock_popen):
        """Test command execution with stderr output."""
        # Setup mocks
        mock_time.side_effect = [0.0, 2.0]
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        # Execute
        result = _execute(
            command_to_run=["false"],
            working_dir=Path("/tmp"),
            timeout=10,
        )

        # Assert
        assert result.output == ""
        assert result.return_code == 1
        assert result.runtime == 2.0
        assert result.stderr == "error message"
        assert result.had_error is True

    @patch("subprocess.Popen")
    def test_execute_command_timeout(self, mock_popen):
        """Test command execution timeout."""
        # Setup mocks
        mock_process = Mock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired(
            "cmd", 5
        )
        mock_popen.return_value = mock_process

        # Execute
        result = _execute(
            command_to_run=["sleep", "10"],
            working_dir=Path("/tmp"),
            timeout=5,
        )

        # Assert
        assert result.output == ""
        assert result.stderr == ""
        assert result.return_code == 0
        assert result.runtime == 5
        assert result.timed_out is True
        mock_process.kill.assert_called_once()

    @patch("subprocess.Popen")
    def test_execute_command_with_string_stdin(self, mock_popen):
        """Test command execution with string stdin."""
        # Setup mocks
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Execute
        result = _execute(
            command_to_run=["cat"],
            working_dir=Path("/tmp"),
            timeout=10,
            stdin="input text",
        )

        # Assert
        mock_process.communicate.assert_called_once_with(
            input=b"input text", timeout=10
        )

    @patch("subprocess.Popen")
    def test_execute_command_with_list_stdin(self, mock_popen):
        """Test command execution with list stdin."""
        # Setup mocks
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Execute
        _ = _execute(
            command_to_run=["cat"],
            working_dir=Path("/tmp"),
            timeout=10,
            stdin=["line1", "line2"],
        )

        # Assert
        mock_process.communicate.assert_called_once_with(
            input=b"line1\nline2", timeout=10
        )

    @patch("subprocess.Popen")
    def test_execute_command_unexpected_error(self, mock_popen):
        """Test command execution with unexpected error."""
        # Setup mocks
        mock_process = Mock()
        mock_process.communicate.side_effect = Exception("Unexpected error")
        mock_popen.return_value = mock_process

        # Execute
        result = _execute(
            command_to_run=["invalid"],
            working_dir=Path("/tmp"),
            timeout=10,
        )

        # Assert
        assert result.output == ""
        assert result.stderr == "Unexpected error"
        assert result.return_code == -1
        assert result.runtime == -1
        assert result.timed_out is False
        assert result.had_unexpected_error is True
        mock_process.kill.assert_called_once()


class TestExecuteSubprocess:
    """Test cases for the execute_subprocess function."""

    @pytest.mark.asyncio
    async def test_execute_subprocess_single_command(self):
        """Test executing subprocess with single command."""
        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            # Setup
            mock_result = SubprocessResult(
                output="hello world",
                return_code=0,
                runtime=1.0,
                timed_out=False,
                stderr="",
            )
            mock_execute.return_value = mock_result

            executable = SubprocessExecutable(
                files={"test.txt": "test content"},
                commands=[Command(command=["echo", "hello"], timeout=10)],
            )

            # Execute
            result = await execute_subprocess_async(executable)

            # Assert
            assert isinstance(result, SubprocessExecutableResult)
            assert len(result.results) == 1
            assert result.results[0] == mock_result
            assert result.elapsed > 0
            assert result.tracked_files == {}

    @pytest.mark.asyncio
    async def test_execute_subprocess_multiple_commands(self):
        """Test executing subprocess with multiple commands."""
        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            # Setup
            mock_results = [
                SubprocessResult(
                    output="result1",
                    return_code=0,
                    runtime=1.0,
                    timed_out=False,
                    stderr="",
                ),
                SubprocessResult(
                    output="result2",
                    return_code=0,
                    runtime=1.5,
                    timed_out=False,
                    stderr="",
                ),
            ]
            mock_execute.side_effect = mock_results

            executable = SubprocessExecutable(
                files={"test.py": "print('hello')"},
                commands=[
                    Command(command=["python", "test.py"], timeout=10),
                    Command(command=["ls", "-la"], timeout=5),
                ],
            )

            # Execute
            result = await execute_subprocess_async(executable)

            # Assert
            assert len(result.results) == 2
            assert result.results == mock_results
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_subprocess_early_stopping(self):
        """Test early stopping on error."""
        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            # Setup - first command fails
            mock_results = [
                SubprocessResult(
                    output="",
                    return_code=1,
                    runtime=1.0,
                    timed_out=False,
                    stderr="error",
                )
            ]
            mock_execute.side_effect = mock_results

            executable = SubprocessExecutable(
                files={},
                commands=[
                    Command(command=["false"], timeout=10),
                    Command(command=["echo", "should not run"], timeout=10),
                ],
                early_stopping=True,
            )

            # Execute
            result = await execute_subprocess_async(executable)

            # Assert
            assert len(result.results) == 1
            assert mock_execute.call_count == 1
            assert result.results[0].had_error is True

    @pytest.mark.asyncio
    async def test_execute_subprocess_no_early_stopping(self):
        """Test continuing execution without early stopping."""
        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            # Setup - first command fails but execution continues
            mock_results = [
                SubprocessResult(
                    output="",
                    return_code=1,
                    runtime=1.0,
                    timed_out=False,
                    stderr="error",
                ),
                SubprocessResult(
                    output="success",
                    return_code=0,
                    runtime=1.0,
                    timed_out=False,
                    stderr="",
                ),
            ]
            mock_execute.side_effect = mock_results

            executable = SubprocessExecutable(
                files={},
                commands=[
                    Command(command=["false"], timeout=10),
                    Command(command=["echo", "hello"], timeout=10),
                ],
                early_stopping=False,
            )

            # Execute
            result = await execute_subprocess_async(executable)

            # Assert
            assert len(result.results) == 2
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_subprocess_tracked_files(self):
        """Test tracking files after execution."""
        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            # Setup
            mock_execute.return_value = SubprocessResult(
                output="",
                return_code=0,
                runtime=1.0,
                timed_out=False,
                stderr="",
            )

            executable = SubprocessExecutable(
                files={"input.txt": "input content"},
                commands=[
                    Command(
                        command=["cp", "input.txt", "output.txt"], timeout=10
                    )
                ],
                tracked_files=["output.txt"],
            )

            # Execute
            with patch("pathlib.Path.read_text") as mock_read:
                mock_read.return_value = "output content"
                result = await execute_subprocess_async(executable)

            # Assert
            assert "output.txt" in result.tracked_files
            assert result.tracked_files["output.txt"] == "output content"

    @pytest.mark.asyncio
    async def test_execute_subprocess_file_writing(self, tmpdir):
        """Test that files are written to temp directory."""

        with patch(
            "code_execution.executables.subproc._execute"
        ) as mock_execute:
            with patch(
                "code_execution.executables.subproc.aiofiles.open"
            ) as mock_write:
                # Setup
                mock_execute.return_value = SubprocessResult(
                    output="",
                    return_code=0,
                    runtime=1.0,
                    timed_out=False,
                    stderr="",
                )

                executable = SubprocessExecutable(
                    files={
                        "test1.py": "print('hello')",
                        "test2.txt": "text content",
                    },
                    commands=[Command(command=["ls"], timeout=10)],
                )

                # Execute
                _ = await execute_subprocess_async(executable)

                # Assert
                assert mock_write.call_count == 2
