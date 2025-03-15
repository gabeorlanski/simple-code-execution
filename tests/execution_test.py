import logging
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from code_execution import execution
from code_execution.configs import ExecutionConfig
from code_execution.data_structures import Command
from code_execution.data_structures import CommandsToRun


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


@pytest.fixture()
def dummy_commands(request):
    yield [
        Command(command=["python", "test.py"], timeout=1),
    ] * request.param


def make_cmd_to_run(prog, cwd, commands=None):
    cwd = Path(cwd)

    with cwd.joinpath("test.py").open("w") as f:
        f.write(prog)

    if commands is None:
        commands = [Command(command=["python", "test.py"], timeout=1)]

    return CommandsToRun(
        cwd=cwd,
        commands=commands,
        tracked_files=["test.py"],
    )


@pytest.fixture()
def pass_cmd_to_run(passing_program, tmpdir, request):
    cmds = [Command(command=["python", "test.py"], timeout=1)] * request.param
    yield make_cmd_to_run(passing_program, tmpdir, cmds)


@pytest.mark.parametrize("pass_cmd_to_run", [1, 2], indirect=True)
def test_execute_code(passing_program, pass_cmd_to_run):
    result = execution.serial_execute_code(key="1.0", sample=pass_cmd_to_run)
    assert not result.had_error
    assert not result.timed_out
    assert len(result.command_results) == len(pass_cmd_to_run.commands)
    assert all(r.return_code == 0 for r in result.command_results)
    assert result.tracked_files == {
        "test.py": passing_program,
    }


@pytest.mark.parametrize("num_commands", [1, 2])
def test_execute_code_fail(error_program, tmpdir, num_commands):
    cmd = [Command(command=["python", "test.py"], timeout=1)] * num_commands
    cmd_to_run = make_cmd_to_run(error_program, tmpdir, commands=cmd)

    result = execution.serial_execute_code(key="1.0", sample=cmd_to_run)

    assert result.key == "1.0"
    assert result.had_error
    assert not result.timed_out
    assert len(result.command_results) == 1
    assert result.command_results[0].return_code != 0
    assert result.tracked_files == {
        "test.py": error_program,
    }


@pytest.mark.parametrize("dummy_commands", [1, 2], indirect=True)
def test_execute_code_timeout(timeout_program, tmpdir, dummy_commands):
    cmd_to_run = make_cmd_to_run(
        timeout_program, tmpdir, commands=dummy_commands
    )

    result = execution.serial_execute_code(key="1.0", sample=cmd_to_run)
    assert result.key == "1.0"
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
                "executable": CommandsToRun(
                    commands=[Command(command=command, timeout=2)],
                    cwd=pred_dir,
                    tracked_files=[],
                ),
            }
        )

    config = ExecutionConfig(
        num_workers=4,
        batch_size=batch_size,
        num_executors=num_executors,
    )

    *_, results = execution.execute_commands(commands, config)
    keys, results = zip(*results)
    assert [i[0] for i in sorted(keys, key=lambda x: x[0])] == list(
        range(num_preds)
    )
    assert len(results) == len(predictions)
    assert all(not r.had_error for r in results)
    assert all(r.last_cmd.return_code == 0 for r in results)


def make_ignore_error_test(
    passing_program, failing_program, cwd, ignore_errors, cmds
):
    cwd = Path(cwd)
    with cwd.joinpath("fail.py").open("w") as f:
        f.write(failing_program)
    with cwd.joinpath("pass.py").open("w") as f:
        f.write(passing_program)

    pass_cmd = Command(
        command=["python", "pass.py"],
        timeout=1,
        ignore_error=ignore_errors == "cmd",
    )
    fail_cmd = Command(
        command=["python", "fail.py"],
        timeout=1,
        ignore_error=ignore_errors == "cmd",
    )
    commands = []
    first_fail = None
    for command in cmds:
        if command == "F":
            commands.append(fail_cmd)
            if ignore_errors == "none" and first_fail is None:
                first_fail = len(commands)
        else:
            commands.append(pass_cmd)
    first_fail = first_fail or len(commands)
    command_to_run = CommandsToRun(
        cwd=cwd,
        commands=commands,
        tracked_files=[],
        ensure_all_run=ignore_errors == "all",
    )

    return first_fail, command_to_run


@pytest.mark.parametrize("ignore_errors", ["all", "cmd", "none"])
@pytest.mark.parametrize("cmds", ["PPP", "FFF", "PFP", "F", "P", "FPP"])
def test_execute_ignore_errors(
    passing_program, error_program, tmpdir, ignore_errors, cmds
):
    first_fail, to_run = make_ignore_error_test(
        passing_program, error_program, tmpdir, ignore_errors, cmds
    )

    result = execution.serial_execute_code(key="1.0", sample=to_run)

    assert len(result.command_results) == first_fail
    for actual, expected in zip(result.command_results, cmds):
        if expected == "P":
            assert actual.return_code == 0
        else:
            assert actual.return_code != 0
            assert not actual.had_unexpected_error


@pytest.mark.parametrize("as_list", [True, False])
def test_execute_stdin(stdin_program, tmpdir, as_list):
    cwd = Path(tmpdir)
    stdin = ["1", "2", "4"]
    if not as_list:
        stdin = "\n".join(stdin)
    command = make_command(stdin_program, cwd)
    command = CommandsToRun(
        cwd=cwd,
        commands=[Command(command=command, timeout=1, stdin=stdin)],
    )
    result = execution.serial_execute_code(key="1.0", sample=command)

    assert result.key == "1.0"
    assert result.command_results[0].stdout == "Input 1: 1\nInput 2: 2\n"


@pytest.mark.parametrize("as_list", [True, False])
def test_execute_looped_stdin(loop_stdin_program, tmpdir, as_list):
    cwd = Path(tmpdir)
    stdin = ["1", "2", "4"]
    if not as_list:
        stdin = "\n".join(stdin)
    command = make_command(loop_stdin_program, cwd)
    command = CommandsToRun(
        cwd=cwd,
        commands=[Command(command=command, timeout=1, stdin=stdin)],
    )
    result = execution.serial_execute_code(key=1.0, sample=command)

    assert result.key == 1.0
    assert result.command_results[0].stdout == "Input: 1\nInput: 2\nInput: 4\n"
    assert result.command_results[0].stderr.endswith(
        "EOFError: EOF when reading a line\n"
    )
