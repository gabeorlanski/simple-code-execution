"""Unit tests for the FastAPI server."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from code_execution.executables.base import Command
from code_execution.executables.base import ExecutableResult
from code_execution.executables.subproc import SubprocessResult
from code_execution.server.main import app
from code_execution.server.models import ExecuteRequest

client = TestClient(app)


class TestHealthEndpoint:
    """Test cases for the health endpoint."""

    def test_health_check(self):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestExecuteEndpoint:
    """Test cases for the execute endpoint."""

    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_successful_request(self, mock_execute):
        """Test successful code execution."""
        # Setup mock
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="Hello World!\n",
                    return_code=0,
                    runtime=0.1,
                    timed_out=False,
                    stderr=""
                )
            ],
            elapsed=0.12,
            tracked_files={}
        )
        mock_metadata = {"execution_id": "exec_1", "queue_info": {}}
        mock_execute.return_value = (mock_result, mock_metadata)

        # Make request
        request_data = {
            "files": {"hello.py": "print('Hello World!')"},
            "commands": [{"command": ["python", "hello.py"], "timeout": 30}],
            "executable_type": "subprocess",
            "early_stopping": False,
            "tracked_files": []
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "result" in response_data
        assert len(response_data["result"]["results"]) == 1
        assert response_data["result"]["results"][0]["output"] == "Hello World!\n"
        assert response_data["result"]["elapsed"] == 0.12

    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_with_tracked_files(self, mock_execute):
        """Test execution with tracked files."""
        # Setup mock
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="Script completed\n",
                    return_code=0,
                    runtime=0.05,
                    timed_out=False,
                    stderr=""
                )
            ],
            elapsed=0.06,
            tracked_files={"output.txt": "Generated content"}
        )
        mock_metadata = {"execution_id": "exec_2", "queue_info": {}}
        mock_execute.return_value = (mock_result, mock_metadata)

        # Make request
        request_data = {
            "files": {"script.py": "with open('output.txt', 'w') as f: f.write('Generated content')"},
            "commands": [{"command": ["python", "script.py"], "timeout": 30}],
            "executable_type": "subprocess",
            "early_stopping": False,
            "tracked_files": ["output.txt"]
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["result"]["tracked_files"]["output.txt"] == "Generated content"

    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_validation_error(self, mock_execute):
        """Test execution with validation error."""
        # Setup mock to raise ValueError
        mock_execute.side_effect = ValueError("At least one command is required")

        # Make request with invalid data
        request_data = {
            "files": {"test.py": "print('test')"},
            "commands": [],  # Empty commands should cause validation error
            "executable_type": "subprocess",
            "early_stopping": False,
            "tracked_files": []
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert error response
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data
        assert "At least one command is required" in response_data["details"]

    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_runtime_error(self, mock_execute):
        """Test execution with runtime error."""
        # Setup mock to raise RuntimeError
        mock_execute.side_effect = RuntimeError("Execution failed: Command not found")

        # Make request
        request_data = {
            "files": {},
            "commands": [{"command": ["nonexistent_command"], "timeout": 30}],
            "executable_type": "subprocess",
            "early_stopping": False,
            "tracked_files": []
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert error response
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data
        assert "Execution failed" in response_data["details"]

    def test_execute_invalid_json(self):
        """Test execution with invalid JSON data."""
        response = client.post(
            "/execute",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_execute_missing_required_fields(self):
        """Test execution with missing required fields."""
        request_data = {
            "files": {"test.py": "print('test')"}
            # Missing commands field
        }
        
        response = client.post("/execute", json=request_data)
        
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data

    def test_get_queue_status(self):
        """Test getting queue status."""
        response = client.get("/queue/status")
        
        assert response.status_code == 200
        response_data = response.json()
        assert "current_executions" in response_data
        assert "max_concurrency" in response_data
        assert "queue_size" in response_data
        assert "available_slots" in response_data


    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_with_default_values(self, mock_execute):
        """Test execution with default values."""
        # Setup mock
        mock_result = ExecutableResult(
            results=[
                SubprocessResult(
                    output="Success\n",
                    return_code=0,
                    runtime=0.01,
                    timed_out=False,
                    stderr=""
                )
            ],
            elapsed=0.02,
            tracked_files={}
        )
        mock_metadata = {"execution_id": "exec_3", "queue_info": {}}
        mock_execute.return_value = (mock_result, mock_metadata)

        # Make minimal request (should use defaults)
        request_data = {
            "files": {"test.py": "print('Success')"},
            "commands": [{"command": ["python", "test.py"]}]  # No timeout specified
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        
        # Verify the service was called with defaults
        mock_execute.assert_called_once()
        called_request = mock_execute.call_args[0][0]
        assert called_request.executable_type == "subprocess"
        assert called_request.early_stopping is False
        assert called_request.tracked_files == []
        assert called_request.priority == 5  # default priority

    @patch("code_execution.server.service.ExecutionService.execute_request")
    def test_execute_unexpected_error(self, mock_execute):
        """Test execution with unexpected error."""
        # Setup mock to raise unexpected exception
        mock_execute.side_effect = Exception("Unexpected system error")

        # Make request
        request_data = {
            "files": {"test.py": "print('test')"},
            "commands": [{"command": ["python", "test.py"], "timeout": 30}],
            "executable_type": "subprocess",
            "early_stopping": False,
            "tracked_files": []
        }
        
        response = client.post("/execute", json=request_data)
        
        # Assert error response
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data["detail"]
        assert response_data["detail"]["error"] == "Internal Server Error"
        assert response_data["detail"]["details"] == "An unexpected error occurred"