import functools
import json
from collections import defaultdict
from pathlib import Path
from unittest import mock

import pytest

from code_execution import Executable
from code_execution import ExecutionConfig
from code_execution import entrypoints
from code_execution import safe_ast_parse
from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import ExecutionResult


@pytest.fixture()
def execution_config():
    yield ExecutionConfig(
        num_workers=2,
    )


def dummy_proc(i, num_returns):
    if num_returns == 1:
        return Executable(
            files={"test": i},
            commands=[Command(command=0, timeout=1)],
            tracked_files=[],
        )
    return [
        Executable(
            files={
                "test": i,
            },
            commands=[Command(command=j, timeout=1)],
            tracked_files=[],
        )
        for j in range(num_returns)
    ]


def _make_dummy_execution_result(stdout):
    return ExecutionResult(
        key="test",
        command_results=[
            CommandResult(
                return_code=1,
                timed_out=False,
                stdout=stdout,
                stderr="",
                runtime=0.0,
                had_unexpected_error=False,
            )
        ],
        elapsed=0.0,
        cwd="",
        tracked_files={},
        expected_num_commands=1,
    )


def dummy_proc_filtered(i):
    if i == 0:
        return [
            Executable(
                files={
                    "test": i,
                },
                commands=[{"command": 0, "timeout": 1}],
                tracked_files=[],
            ),
            _make_dummy_execution_result("0"),
        ]
    elif i == 2:
        return [
            _make_dummy_execution_result("1"),
            Executable(
                files={
                    "test": i,
                },
                commands=[{"command": 0, "timeout": 1}],
                tracked_files=[],
            ),
        ]
    return [
        Executable(
            files={
                "test": i,
            },
            commands=[{"command": j, "timeout": 1}],
            tracked_files=[],
        )
        for j in range(2)
    ]


@pytest.mark.parametrize("num_returns", [1, 2], ids=["single", "multiple"])
@pytest.mark.parametrize("num_workers", [1, 2], ids=["serial", "parallel"])
def test_preproc(execution_config, num_returns, num_workers, tmpdir):
    cwd = Path(tmpdir)
    processor = functools.partial(dummy_proc, num_returns=num_returns)

    expected_files = []
    expected_commands = []
    for i in range(10):
        for j in range(num_returns):
            expected_files.append((f"{i}.{j}", {"test": i}))
            expected_commands.append(
                {
                    "idx": i,
                    "sub_idx": j,
                    "commands": [Command(command=j, timeout=1)],
                }
            )

    execution_config.num_workers = num_workers
    files_to_write, commands, filtered_out, timings = (
        entrypoints.preprocess_commands(
            config=execution_config,
            dir_to_use=cwd,
            pred_list=list(range(10)),
            preprocessor=processor,
            preproc_returns_list=num_returns > 1,
        )
    )

    assert filtered_out == {}
    assert len(files_to_write) == len(expected_files)
    assert len(commands) == len(expected_commands)
    assert set(timings.keys()) == {
        f"{i}.{j}" for i in range(10) for j in range(num_returns)
    }
    for actual, expected in zip(files_to_write, expected_files):
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]

    for actual, expected in zip(commands, expected_commands):
        assert actual["key"] == (expected["idx"], expected["sub_idx"])
        assert actual["executable"].commands == expected["commands"]


@pytest.mark.parametrize("num_returns", [1, 2], ids=["single", "multiple"])
@pytest.mark.parametrize("num_workers", [1, 2], ids=["serial", "parallel"])
def test_preproc_batched(execution_config, num_returns, num_workers, tmpdir):
    cwd = Path(tmpdir)
    processor = functools.partial(dummy_proc, num_returns=num_returns)

    expected_files = []
    expected_commands = []
    for i in range(10):
        for j in range(num_returns):
            expected_files.append((f"{i}.{j}", {"test": i}))
            expected_commands.append(
                {
                    "idx": i,
                    "sub_idx": j,
                    "commands": [Command(command=j, timeout=1)],
                }
            )

    execution_config.num_workers = num_workers
    files_to_write, commands, filtered_out, timings = (
        entrypoints.preprocess_commands(
            config=execution_config,
            dir_to_use=cwd,
            pred_list=list(range(10)),
            preprocessor=processor,
            preproc_returns_list=num_returns > 1,
            batch_size=2,
        )
    )
    assert filtered_out == {}
    assert len(files_to_write) == len(expected_files)
    assert len(commands) == len(expected_commands)
    assert set(timings.keys()) == {
        f"{i}.{j}" for i in range(10) for j in range(num_returns)
    }
    for actual, expected in zip(files_to_write, expected_files):
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]

    for actual, expected in zip(commands, expected_commands):
        assert actual["key"] == (expected["idx"], expected["sub_idx"])
        assert actual["executable"].commands == expected["commands"]


def test_preproc_filtered_out(execution_config, tmpdir):
    cwd = Path(tmpdir)
    files_to_write, commands, filtered_out, timings = (
        entrypoints.preprocess_commands(
            config=execution_config,
            dir_to_use=cwd,
            pred_list=list(range(10)),
            preprocessor=dummy_proc_filtered,
            preproc_returns_list=True,
            batch_size=2,
        )
    )

    assert len(files_to_write) == 18
    assert len(commands) == 18
    command_keys = {c["key"] for c in commands}
    expected_filtered_keys = {(0, 1), (2, 0)}
    assert not any(k in command_keys for k in expected_filtered_keys)
    assert len(filtered_out) == 2
    assert set(filtered_out.keys()) == {(0, 1), (2, 0)}
    assert filtered_out[(0, 1)].last_cmd.stdout == "0"
    assert filtered_out[(2, 0)].last_cmd.stdout == "1"
    assert len(timings) == 20
