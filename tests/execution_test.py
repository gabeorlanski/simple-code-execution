import logging
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from code_execution import execution
from code_execution.configs import ExecutionConfig


def make_command(program, cwd: Path):
    with cwd.joinpath("test.py").open("w") as f:
        f.write(program)
    return ["python", "test.py"]


def test_safe_execute(passing_program, pass_print, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(passing_program, cwd)

    result = execution.safe_execute(command, cwd, timeout=1)
    assert result.return_code == 0
    assert not result.stderr
    assert result.stdout == pass_print + "\n"
    assert not result.timed_out
    assert not result.had_unexpected_error


def test_safe_execute_timeout(timeout_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(timeout_program, cwd)

    result = execution.safe_execute(command, cwd, timeout=1)
    assert result.timed_out
    assert not result.had_unexpected_error
    assert result.runtime >= 1


def test_safe_execute_fail(error_program, fail_print, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(error_program, cwd)

    result = execution.safe_execute(command, cwd, timeout=1)
    assert result.return_code != 0
    assert fail_print in result.stderr
    assert not result.timed_out
    assert not result.had_unexpected_error
    assert result.stdout == fail_print + "\n"


def test_execute_code(passing_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(passing_program, cwd)

    exec_dict = {
        "cwd": cwd,
        "commands": [
            {
                "command": command,
                "timeout": 1,
            },
            {
                "command": command,
                "timeout": 1,
            },
        ],
        "tracked_files": ["test.py"],
    }

    result = execution.serial_execute_code(exec_dict)
    assert not result.had_error
    assert not result.timed_out
    assert len(result.command_results) == 2
    assert result.command_results[0].return_code == 0
    assert result.command_results[1].return_code == 0
    assert result.tracked_files == {
        "test.py": passing_program,
    }


def test_execute_code_fail(error_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(error_program, cwd)

    exec_dict = {
        "cwd": cwd,
        "commands": [
            {
                "command": command,
                "timeout": 1,
            },
            {
                "command": command,
                "timeout": 1,
            },
        ],
        "tracked_files": ["test.py"],
    }

    result = execution.serial_execute_code(exec_dict)
    assert result.had_error
    assert not result.timed_out
    assert len(result.command_results) == 1
    assert result.command_results[0].return_code != 0
    assert result.tracked_files == {
        "test.py": error_program,
    }


def test_execute_code_timeout(timeout_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(timeout_program, cwd)

    exec_dict = {
        "cwd": cwd,
        "commands": [
            {
                "command": command,
                "timeout": 1,
            },
            {
                "command": command,
                "timeout": 1,
            },
        ],
        "tracked_files": ["test.py"],
    }

    result = execution.serial_execute_code(exec_dict)
    assert not result.had_error
    assert result.timed_out
    assert len(result.command_results) == 1
    assert result.command_results[0].return_code == 0
    assert result.tracked_files == {
        "test.py": timeout_program,
    }


@pytest.mark.parametrize("num_times", [1, 2, 3])
def test_execute_multiple_times(num_times, passing_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(passing_program, cwd)

    rtr_values = [
        dict(
            return_code=0,
            runtime=i + 1,
            stderr=str(i),
            stdout=str(i),
            timed_out=False,
            had_unexpected_error=False,
        )
        for i in range(num_times)
    ]

    with mock.patch("code_execution.execution._execute") as mock_execute:
        mock_execute.side_effect = rtr_values
        result = execution.safe_execute(
            command, cwd, timeout=1, num_times=num_times
        )
        assert len(mock_execute.call_args_list) == num_times
        assert result.return_code == 0
        assert math.isclose(
            result.runtime, np.mean(list(range(1, num_times + 1)))
        )
        assert result.stderr == f"{num_times-1}"
        assert result.stdout == f"{num_times-1}"


@pytest.mark.parametrize(
    "fail_type", ["return_code", "timed_out", "had_unexpected_error"]
)
def test_execute_multiple_times_fail(fail_type, passing_program, tmpdir):
    cwd = Path(tmpdir)
    command = make_command(passing_program, cwd)

    rtr_code = 1 if fail_type == "return_code" else 0
    timed_out = fail_type == "timed_out"
    had_unexpected_error = fail_type == "had_unexpected_error"

    rtr_values = [
        dict(
            return_code=rtr_code,
            runtime=i + 1,
            stderr=str(i),
            stdout=str(i),
            timed_out=timed_out,
            had_unexpected_error=had_unexpected_error,
        )
        for i in range(4)
    ]

    with mock.patch("code_execution.execution._execute") as mock_execute:
        mock_execute.side_effect = rtr_values
        result = execution.safe_execute(command, cwd, timeout=1, num_times=4)
        assert len(mock_execute.call_args_list) == 1
        assert result.return_code == rtr_code
        assert result.timed_out == timed_out
        assert result.had_unexpected_error == had_unexpected_error
        assert math.isclose(result.runtime, 1.0)
        assert result.stderr == "0"
        assert result.stdout == "0"


@pytest.mark.parametrize("num_executors", [1, 2], ids=["Single", "Multiple"])
@pytest.mark.parametrize("batch_size", [1, 4], ids=["bs_1", "bs_4"])
def test_parallel_code_execution(
    passing_program, tmpdir, num_executors, batch_size, caplog
):
    caplog.set_level(logging.DEBUG)
    cwd = Path(tmpdir)
    num_preds = 100
    predictions = [passing_program for _ in range(num_preds)]
    commands = []
    for i in range(num_preds):
        pred_dir = cwd / f"pred_{i}"
        pred_dir.mkdir()
        command = make_command(predictions[i], pred_dir)
        commands.append(
            {
                "key": (i, 0),
                "executable": {
                    "commands": [{"command": command, "timeout": 2}],
                    "cwd": pred_dir,
                    "tracked_files": [],
                },
            }
        )

    config = ExecutionConfig(
        num_workers=4,
        batch_size=batch_size,
        num_executors=num_executors,
    )

    results = execution.execute_commands(commands, config)
    keys, results = zip(*results)
    assert [i[0] for i in sorted(keys, key=lambda x: x[0])] == list(
        range(num_preds)
    )
    assert len(results) == len(predictions)
    assert all(not r.had_error for r in results)
    assert all(r.last_cmd.return_code == 0 for r in results)
