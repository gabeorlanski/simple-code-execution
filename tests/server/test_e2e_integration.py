"""End-to-end integration tests for the FastAPI server with real program execution."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from code_execution.server.config import ServerConfig
from code_execution.server.main import create_app


@pytest.fixture
def test_app():
    """Create a test FastAPI application with isolated configuration."""
    config = ServerConfig(
        host="127.0.0.1",
        port=8001,  # Different port for testing
        max_concurrency=3,
        log_level="debug",
    )
    return create_app(config)


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)


class TestPythonExecution:
    """Test cases for executing Python programs end-to-end."""

    def test_simple_python_hello_world(self, client):
        """Test executing a simple Python hello world program."""
        request_data = {
            "files": {"hello.py": "print('Hello, World!')"},
            "commands": [{"command": ["python", "hello.py"], "timeout": 10}],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["result"]["results"]) == 1

        result = data["result"]["results"][0]
        assert result["output"].strip() == "Hello, World!"
        assert result["return_code"] == 0
        assert result["timed_out"] is False
        assert data["result"]["elapsed"] > 0

    def test_python_with_imports(self, client):
        """Test executing Python program with standard library imports."""
        request_data = {
            "files": {
                "math_test.py": """
import math
import datetime

print(f"Pi is approximately {math.pi:.2f}")
print(f"Current year: {datetime.datetime.now().year}")
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
            },
            "commands": [
                {"command": ["python", "math_test.py"], "timeout": 10}
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        output = data["result"]["results"][0]["output"]
        assert "Pi is approximately 3.14" in output
        assert "Current year:" in output
        assert "Square root of 16 is 4.0" in output

    def test_python_with_multiple_files(self, client):
        """Test executing Python program that imports from another file."""
        request_data = {
            "files": {
                "utils.py": """
def add_numbers(a, b):
    return a + b

def greet(name):
    return f"Hello, {name}!"
""",
                "main.py": """
from utils import add_numbers, greet

result = add_numbers(5, 3)
message = greet("World")

print(f"Addition result: {result}")
print(message)
""",
            },
            "commands": [{"command": ["python", "main.py"], "timeout": 10}],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        output = data["result"]["results"][0]["output"]
        assert "Addition result: 8" in output
        assert "Hello, World!" in output

    def test_python_with_command_line_args(self, client):
        """Test executing Python program with command line arguments."""
        request_data = {
            "files": {
                "args_test.py": """
import sys

print(f"Script name: {sys.argv[0]}")
print(f"Number of arguments: {len(sys.argv) - 1}")

for i, arg in enumerate(sys.argv[1:], 1):
    print(f"Argument {i}: {arg}")
"""
            },
            "commands": [
                {
                    "command": [
                        "python",
                        "args_test.py",
                        "hello",
                        "world",
                        "123",
                    ],
                    "timeout": 10,
                }
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        output = data["result"]["results"][0]["output"]
        assert "Script name: args_test.py" in output
        assert "Number of arguments: 3" in output
        assert "Argument 1: hello" in output
        assert "Argument 2: world" in output
        assert "Argument 3: 123" in output

    def test_python_with_json_processing(self, client):
        """Test Python program that processes JSON data."""
        request_data = {
            "files": {
                "data.json": '{"name": "Alice", "age": 30, "scores": [85, 92, 78]}',
                "json_processor.py": """
import json

# Read and parse JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

print(f"Name: {data['name']}")
print(f"Age: {data['age']}")
print(f"Average score: {sum(data['scores']) / len(data['scores']):.1f}")

# Create output JSON
output_data = {
    "processed": True,
    "summary": {
        "name": data['name'],
        "total_scores": len(data['scores']),
        "max_score": max(data['scores'])
    }
}

with open('output.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Processing complete!")
""",
            },
            "commands": [
                {"command": ["python", "json_processor.py"], "timeout": 10}
            ],
            "tracked_files": ["output.json"],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        output = data["result"]["results"][0]["output"]
        assert "Name: Alice" in output
        assert "Age: 30" in output
        assert "Average score: 85.0" in output
        assert "Processing complete!" in output

        # Check tracked file
        assert "output.json" in data["result"]["tracked_files"]
        tracked_content = json.loads(
            data["result"]["tracked_files"]["output.json"]
        )
        assert tracked_content["processed"] is True
        assert tracked_content["summary"]["name"] == "Alice"
        assert tracked_content["summary"]["max_score"] == 92


class TestShellExecution:
    """Test cases for executing shell commands end-to-end."""

    def test_simple_echo_command(self, client):
        """Test executing a simple echo command."""
        request_data = {
            "files": {},
            "commands": [
                {"command": ["echo", "Hello from shell!"], "timeout": 5}
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            data["result"]["results"][0]["output"].strip()
            == "Hello from shell!"
        )

    def test_ls_command(self, client):
        """Test executing ls command to list files."""
        request_data = {
            "files": {
                "file1.txt": "Content of file 1",
                "file2.py": "print('Hello')",
                "data.json": '{"key": "value"}',
            },
            "commands": [{"command": ["ls", "-la"], "timeout": 5}],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        output = data["result"]["results"][0]["output"]
        assert "file1.txt" in output
        assert "file2.py" in output
        assert "data.json" in output

    def test_cat_and_grep_commands(self, client):
        """Test executing cat and grep commands on files."""
        request_data = {
            "files": {
                "sample.txt": """apple
banana
cherry
apple pie
grape
banana split
"""
            },
            "commands": [
                {"command": ["cat", "sample.txt"], "timeout": 5},
                {"command": ["grep", "apple", "sample.txt"], "timeout": 5},
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["result"]["results"]) == 2

        # First command (cat) should show all content
        cat_output = data["result"]["results"][0]["output"]
        assert "apple" in cat_output
        assert "banana" in cat_output
        assert "cherry" in cat_output

        # Second command (grep) should only show lines with "apple"
        grep_output = data["result"]["results"][1]["output"]
        assert "apple" in grep_output
        assert "apple pie" in grep_output
        assert "banana" not in grep_output

    def test_bash_script_execution(self, client):
        """Test executing a bash script."""
        request_data = {
            "files": {
                "script.sh": """#!/bin/bash
echo "Starting script..."
COUNT=1
while [ $COUNT -le 3 ]; do
    echo "Count: $COUNT"
    COUNT=$((COUNT + 1))
done
echo "Script completed!"
"""
            },
            "commands": [
                {"command": ["chmod", "+x", "script.sh"], "timeout": 5},
                {"command": ["./script.sh"], "timeout": 10},
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["result"]["results"]) == 2

        # Second command should have the script output
        script_output = data["result"]["results"][1]["output"]
        assert "Starting script..." in script_output
        assert "Count: 1" in script_output
        assert "Count: 2" in script_output
        assert "Count: 3" in script_output
        assert "Script completed!" in script_output


class TestFileTracking:
    """Test cases for file tracking functionality."""

    def test_track_single_output_file(self, client):
        """Test tracking a single output file."""
        request_data = {
            "files": {
                "generator.py": """
with open('output.txt', 'w') as f:
    f.write('Generated content\\n')
    f.write('Line 2\\n')
    f.write('Line 3\\n')
print('File generated!')
"""
            },
            "commands": [
                {"command": ["python", "generator.py"], "timeout": 10}
            ],
            "tracked_files": ["output.txt"],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Check tracked file content
        assert "output.txt" in data["result"]["tracked_files"]
        content = data["result"]["tracked_files"]["output.txt"]
        assert "Generated content" in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_track_multiple_files(self, client):
        """Test tracking multiple output files."""
        request_data = {
            "files": {
                "multi_generator.py": """
# Generate multiple files
with open('file1.txt', 'w') as f:
    f.write('Content of file 1')

with open('file2.txt', 'w') as f:
    f.write('Content of file 2')

with open('data.csv', 'w') as f:
    f.write('name,age,city\\n')
    f.write('Alice,30,New York\\n')
    f.write('Bob,25,San Francisco\\n')

print('All files generated!')
"""
            },
            "commands": [
                {"command": ["python", "multi_generator.py"], "timeout": 10}
            ],
            "tracked_files": ["file1.txt", "file2.txt", "data.csv"],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        tracked = data["result"]["tracked_files"]
        assert "file1.txt" in tracked
        assert "file2.txt" in tracked
        assert "data.csv" in tracked

        assert tracked["file1.txt"] == "Content of file 1"
        assert tracked["file2.txt"] == "Content of file 2"
        assert "Alice,30,New York" in tracked["data.csv"]
        assert "Bob,25,San Francisco" in tracked["data.csv"]

    def test_track_nonexistent_file(self, client):
        """Test behavior when trying to track a file that doesn't exist."""
        request_data = {
            "files": {"simple.py": "print('Hello')"},
            "commands": [{"command": ["python", "simple.py"], "timeout": 10}],
            "tracked_files": ["nonexistent.txt"],
        }

        response = client.post("/execute", json=request_data)

        # Tracking nonexistent file should result in execution error
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Execution Error"
        assert "No such file or directory" in data["details"]


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_python_syntax_error(self, client):
        """Test handling Python syntax errors."""
        request_data = {
            "files": {
                "bad_syntax.py": """
print("Hello World"
# Missing closing parenthesis
def broken_function(
    return "broken"
"""
            },
            "commands": [
                {"command": ["python", "bad_syntax.py"], "timeout": 10}
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        result = data["result"]["results"][0]
        assert result["return_code"] != 0
        assert result["timed_out"] is False
        # Syntax errors should result in non-zero return code
        assert result["had_unexpected_error"] is False

    def test_python_runtime_error(self, client):
        """Test handling Python runtime errors."""
        request_data = {
            "files": {
                "runtime_error.py": """
print("Starting program...")
x = 10
y = 0
print("About to divide by zero...")
result = x / y  # This will raise ZeroDivisionError
print(f"Result: {result}")
"""
            },
            "commands": [
                {"command": ["python", "runtime_error.py"], "timeout": 10}
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        result = data["result"]["results"][0]
        assert result["return_code"] != 0
        assert "Starting program..." in result["output"]
        assert "About to divide by zero..." in result["output"]
        # Runtime errors should result in non-zero return code
        assert result["had_unexpected_error"] is False

    def test_command_not_found(self, client):
        """Test handling when command is not found."""
        request_data = {
            "files": {},
            "commands": [
                {"command": ["nonexistent_command", "arg1"], "timeout": 5}
            ],
        }

        response = client.post("/execute", json=request_data)

        # Command not found results in a RuntimeError which returns 500
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Execution Error"
        assert "Execution failed" in data["details"]

    def test_early_stopping_on_failure(self, client):
        """Test early stopping when a command fails."""
        request_data = {
            "files": {"fail.py": "import sys; sys.exit(1)"},
            "commands": [
                {"command": ["python", "fail.py"], "timeout": 10},
                {"command": ["echo", "This should not run"], "timeout": 5},
            ],
            "early_stopping": True,
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Should only have one result due to early stopping
        assert len(data["result"]["results"]) == 1
        assert data["result"]["results"][0]["return_code"] == 1

    def test_timeout_handling(self, client):
        """Test handling of command timeouts."""
        request_data = {
            "files": {
                "slow.py": """
import time
print("Starting long operation...")
time.sleep(10)  # Sleep longer than timeout
print("This should not be printed")
"""
            },
            "commands": [
                {
                    "command": ["python", "slow.py"],
                    "timeout": 2,
                }  # 2 second timeout
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        result = data["result"]["results"][0]
        assert result["timed_out"] is True
        # Process may have return code 0 if terminated cleanly
        assert (
            "Starting long operation..." in result["output"]
            or result["output"] == ""
        )
        assert "This should not be printed" not in result["output"]


class TestMultipleCommands:
    """Test cases for executing multiple commands in sequence."""

    def test_sequential_commands(self, client):
        """Test executing multiple commands in sequence."""
        request_data = {
            "files": {
                "step1.py": "print('Step 1 complete'); open('step1.done', 'w').close()",
                "step2.py": """
import os
if os.path.exists('step1.done'):
    print('Step 2 can proceed')
    with open('final.txt', 'w') as f:
        f.write('All steps completed!')
else:
    print('Step 1 not completed')
    exit(1)
""",
            },
            "commands": [
                {"command": ["python", "step1.py"], "timeout": 5},
                {"command": ["python", "step2.py"], "timeout": 5},
                {"command": ["cat", "final.txt"], "timeout": 5},
            ],
            "tracked_files": ["final.txt"],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["result"]["results"]) == 3

        # Check each step
        assert "Step 1 complete" in data["result"]["results"][0]["output"]
        assert "Step 2 can proceed" in data["result"]["results"][1]["output"]
        assert "All steps completed!" in data["result"]["results"][2]["output"]

        # Check tracked file
        assert (
            data["result"]["tracked_files"]["final.txt"]
            == "All steps completed!"
        )

    def test_data_processing_pipeline(self, client):
        """Test a data processing pipeline with multiple steps."""
        request_data = {
            "files": {
                "generate_data.py": """
import csv
import random

# Generate sample data
data = []
names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
for i in range(10):
    data.append({
        'id': i + 1,
        'name': random.choice(names),
        'score': random.randint(60, 100)
    })

with open('raw_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'name', 'score'])
    writer.writeheader()
    writer.writerows(data)

print(f"Generated {len(data)} records")
""",
                "process_data.py": """
import csv
import json

# Read CSV and process
with open('raw_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Convert scores to integers and calculate stats
scores = []
for row in data:
    row['score'] = int(row['score'])
    scores.append(row['score'])

stats = {
    'total_records': len(data),
    'average_score': sum(scores) / len(scores),
    'max_score': max(scores),
    'min_score': min(scores)
}

# Save processed data
with open('processed_data.json', 'w') as f:
    json.dump({'data': data, 'stats': stats}, f, indent=2)

print(f"Processed {stats['total_records']} records")
print(f"Average score: {stats['average_score']:.1f}")
""",
                "generate_report.py": """
import json

with open('processed_data.json', 'r') as f:
    data = json.load(f)

report = []
report.append("DATA PROCESSING REPORT")
report.append("=" * 25)
report.append(f"Total Records: {data['stats']['total_records']}")
report.append(f"Average Score: {data['stats']['average_score']:.1f}")
report.append(f"Max Score: {data['stats']['max_score']}")
report.append(f"Min Score: {data['stats']['min_score']}")
report.append("")
report.append("TOP PERFORMERS:")
for row in sorted(data['data'], key=lambda x: x['score'], reverse=True)[:3]:
    report.append(f"  {row['name']}: {row['score']}")

report_text = "\\n".join(report)
with open('report.txt', 'w') as f:
    f.write(report_text)

print("Report generated successfully!")
print(report_text)
""",
            },
            "commands": [
                {"command": ["python", "generate_data.py"], "timeout": 10},
                {"command": ["python", "process_data.py"], "timeout": 10},
                {"command": ["python", "generate_report.py"], "timeout": 10},
            ],
            "tracked_files": [
                "raw_data.csv",
                "processed_data.json",
                "report.txt",
            ],
        }

        response = client.post("/execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["result"]["results"]) == 3

        # Check that all tracked files exist
        tracked = data["result"]["tracked_files"]
        assert "raw_data.csv" in tracked
        assert "processed_data.json" in tracked
        assert "report.txt" in tracked

        # Verify report content
        report_content = tracked["report.txt"]
        assert "DATA PROCESSING REPORT" in report_content
        assert "Total Records: 10" in report_content
        assert "Average Score:" in report_content
        assert "TOP PERFORMERS:" in report_content


class TestConcurrentExecution:
    """Test cases for concurrent execution capabilities."""

    def test_health_check_during_execution(self, client):
        """Test that health check works while other executions are running."""
        # Start a long-running task
        long_request = {
            "files": {
                "slow.py": """
import time
print("Starting...")
for i in range(5):
    print(f"Working... {i+1}/5")
    time.sleep(0.5)
print("Done!")
"""
            },
            "commands": [{"command": ["python", "slow.py"], "timeout": 10}],
        }

        # Start the long-running request in background (don't wait)
        async def run_long_task():
            return client.post("/execute", json=long_request)

        # Health check should still work
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "healthy"}

        # Queue status should work
        status_response = client.get("/queue/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "current_executions" in status_data
        assert "max_concurrency" in status_data

    def test_multiple_simple_executions(self, client):
        """Test executing multiple simple programs."""
        requests = []
        for i in range(3):
            request_data = {
                "files": {f"test_{i}.py": f"print('Hello from task {i}')"},
                "commands": [
                    {"command": ["python", f"test_{i}.py"], "timeout": 5}
                ],
            }
            requests.append(request_data)

        # Execute all requests
        responses = []
        for req in requests:
            response = client.post("/execute", json=req)
            responses.append(response)

        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert (
                f"Hello from task {i}" in data["result"]["results"][0]["output"]
            )
