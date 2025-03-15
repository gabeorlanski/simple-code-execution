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


def _preprocessor(pred):
    tree = safe_ast_parse(pred["prediction"])
    if tree is None:
        return ExecutionResult(
            key=pred["pred_id"],
            command_results=[
                CommandResult(
                    return_code=1,
                    timed_out=False,
                    stdout="FAILED_SYNTAX",
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
    return Executable(
        {"main.py": pred["prediction"]},
        [{"command": ["python3", "main.py"], "timeout": 2}],
        [],
    )


def _postprocessor(pred, result):
    return {
        **pred,
        "syntax_error": result.last_cmd.stdout == "FAILED_SYNTAX",
        **result.to_dict(),
    }


@pytest.fixture()
def execution_entrypoint_fixture(passing_program):
    out = [{"prediction": "def", "pred_id": 0}] + [
        {"prediction": passing_program, "pred_id": i + 1} for i in range(2)
    ]
    out.extend([{"prediction": "1,-)", "pred_id": 3}])
    out.extend(
        [
            {"prediction": passing_program, "pred_id": len(out) + i}
            for i in range(3)
        ]
    )
    out.extend([{"prediction": "1,-)", "pred_id": 7}])
    yield out


@pytest.mark.parametrize(
    "max_at_once", [1, 2, -1], ids=["single", "double", "all"]
)
@pytest.mark.parametrize(
    "in_notebook", [True, False], ids=["multiprocessing", "async"]
)
def test_execute_predictions(
    execution_config,
    execution_entrypoint_fixture,
    tmpdir,
    passing_program,
    max_at_once,
    in_notebook,
):
    cwd = Path(tmpdir)
    execution_config.max_execute_at_once = max_at_once
    with mock.patch(
        "code_execution.utils.in_notebook", return_value=in_notebook
    ):
        result = entrypoints.execute_predictions(
            execution_config,
            pred_list=execution_entrypoint_fixture,
            preprocessor=_preprocessor,
            postprocessor=_postprocessor,
            debug_dir=cwd,
        )

    assert len(result.results) == len(execution_entrypoint_fixture)

    pred_ids = []
    syntax_errors = []
    preds = []
    for r in result.results:
        pred_ids.append(r["pred_id"])
        syntax_errors.append(r["syntax_error"])
        preds.append(r["prediction"])

        if r["syntax_error"]:
            assert abs(r["writing_time"] - 0.0) < 0.1
            assert abs(r["cleanup_time"] - 0.0) < 0.1
        else:
            assert r["writing_time"] > 0.0
            assert r["cleanup_time"] > 0.0

    assert pred_ids == list(range(8))
    assert syntax_errors == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    for i in range(8):
        if i in {0, 3, 7}:
            assert preds[i] != passing_program
        else:
            assert preds[i] == passing_program
