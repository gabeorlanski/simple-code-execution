import functools
import json
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
    files_to_write, commands, filtered_out = entrypoints.preprocess_commands(
        config=execution_config,
        dir_to_use=cwd,
        pred_list=list(range(10)),
        preprocessor=processor,
        preproc_returns_list=num_returns > 1,
    )

    assert filtered_out == {}
    assert len(files_to_write) == len(expected_files)
    assert len(commands) == len(expected_commands)
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
    files_to_write, commands, filtered_out = entrypoints.preprocess_commands(
        config=execution_config,
        dir_to_use=cwd,
        pred_list=list(range(10)),
        preprocessor=processor,
        preproc_returns_list=num_returns > 1,
        batch_size=2,
    )
    assert filtered_out == {}
    assert len(files_to_write) == len(expected_files)
    assert len(commands) == len(expected_commands)
    for actual, expected in zip(files_to_write, expected_files):
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]

    for actual, expected in zip(commands, expected_commands):
        assert actual["key"] == (expected["idx"], expected["sub_idx"])
        assert actual["executable"].commands == expected["commands"]


def test_preproc_filtered_out(execution_config, tmpdir):
    cwd = Path(tmpdir)
    files_to_write, commands, filtered_out = entrypoints.preprocess_commands(
        config=execution_config,
        dir_to_use=cwd,
        pred_list=list(range(10)),
        preprocessor=dummy_proc_filtered,
        preproc_returns_list=True,
        batch_size=2,
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


def _preprocessor(pred):
    tree = safe_ast_parse(pred["prediction"])
    if tree is None:
        return ExecutionResult(
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
        )
    return Executable(
        {"main.py": pred["prediction"]},
        [{"command": ["python3", "main.py"], "timeout": 2}],
        [],
    )


def _postprocessor(pred, result):
    return {**pred, "syntax_error": result.last_cmd.stdout == "FAILED_SYNTAX"}


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
        results = entrypoints.execute_predictions(
            execution_config,
            pred_list=execution_entrypoint_fixture,
            preprocessor=_preprocessor,
            postprocessor=_postprocessor,
            debug_dir=cwd,
        )

    assert len(results) == len(execution_entrypoint_fixture)

    pred_ids = []
    syntax_errors = []
    preds = []
    for r in results:
        pred_ids.append(r["pred_id"])
        syntax_errors.append(r["syntax_error"])
        preds.append(r["prediction"])

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


@pytest.mark.parametrize("idx_type", ["int", "str", "tuple"])
def test_error_dir(tmpdir, execution_config, idx_type):

    tmpdir = Path(tmpdir)
    exec_dir = tmpdir.joinpath("exec")
    raw_preds = []
    write_preds = []
    expected_idx = []
    exec_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        with open(exec_dir / f"pred{i}", "w") as f:
            f.write("test")

        idx = i
        if idx_type == "str":
            idx = f"{i}.0"
            expected_idx.append(f"{i}.0")
        elif idx_type == "tuple":
            idx = (i, 0)
            expected_idx.append(f"{i}.0")
        else:
            expected_idx.append(i)
        write_preds.append((idx, {f"pred{i}": "test"}, exec_dir / f"pred{i}"))
        raw_preds.append({"test_key": i})
    with mock.patch(
        "code_execution.entrypoints.write_executables",
        side_effect=FileExistsError("test"),
    ):
        with pytest.raises(FileExistsError):
            entrypoints._write_maybe_save_error_dir(
                config=execution_config,
                raw_preds=raw_preds,
                files=write_preds,
                exec_dir=tmpdir,
                error_directory=tmpdir / "error",
            )

    assert (tmpdir / "error").exists()
    with (tmpdir / "error" / "errors.jsonl").open() as f:
        errors = list(map(json.loads, f))

    assert len(errors) == 10
    for i, e in enumerate(errors):
        assert e == {
            "use_idx": expected_idx[i],
            "files": write_preds[i][1],
            "pred_dir": str(write_preds[i][2]),
        }
