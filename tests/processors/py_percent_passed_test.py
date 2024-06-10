import ast
import math
from pathlib import Path

import pytest

from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import CommandsToRun
from code_execution.execution import serial_execute_code
from code_execution.processors.python import percent_passed
from code_execution.processors.utils import PredictionOutcome
from code_execution.processors.utils import get_prediction_outcome
from code_execution.processors.utils import get_processor
from code_execution.processors.utils import parse_tc_stdout


@pytest.fixture()
def preprocessor():
    return get_processor("py_percent_passed")[0]


@pytest.fixture()
def postprocessor():
    return get_processor("py_percent_passed")[1]


def test_preprocess_base(basic_prediction, preprocessor):
    imports, context, call_code = preprocessor(
        entry_point=basic_prediction["entry_point"],
        test_cases=basic_prediction["test_cases"],
    )

    assert call_code == percent_passed.CALL_CODE.format(
        test_timeout=None, num_benchmark_trials=100
    )
    assert percent_passed.WRAPPER_CONTEXT in context
    expected_runner = "def run_tests(test_timeout:Optional[int]=None):\n    from solution import add_1\n"

    for idx in range(len(basic_prediction["test_cases"])):
        expected_runner += f"\n    test_{idx}(add_1, test_timeout=test_timeout)"
    expected_runner = ast.unparse(ast.parse(expected_runner))
    assert expected_runner in context
    assert imports == percent_passed.DEFAULT_IMPORTS.format(
        module_name="solution", entry_point=basic_prediction["entry_point"]
    )


@pytest.mark.parametrize(
    "tc_imports", [None, "import time"], ids=["no_imports", "with_imports"]
)
def test_preprocess_test_timeout(basic_prediction, preprocessor, tc_imports):
    imports, context, call_code = preprocessor(
        entry_point=basic_prediction["entry_point"],
        test_cases=basic_prediction["test_cases"],
        tc_timeout=0.25,
        tc_imports=tc_imports,
    )

    assert call_code == percent_passed.CALL_CODE.format(
        test_timeout=0.25, num_benchmark_trials=100
    )
    expected_imports = percent_passed.DEFAULT_IMPORTS
    if tc_imports:
        expected_imports += "\n" + tc_imports
    assert imports == expected_imports


@pytest.mark.parametrize("test_1", ["PASSED", "FAILED", "NameError", "MISSING"])
@pytest.mark.parametrize("test_2", ["PASSED", "FAILED", "NameError", "MISSING"])
def test_postprocess_base(test_1, test_2, postprocessor):
    command_result = CommandResult(
        return_code=test_1 == test_2 == "MISSING",
        stdout=f"TEST_1___{test_1}\nTEST_2___{test_2}\n__MEMORY_USED__=100\n__RUNTIME__=0.2",
        stderr="",
        timed_out=False,
        runtime=0.2,
    )

    expected_parse_result = parse_tc_stdout(
        command_result.stdout, expected_test_cases={"1", "2"}
    )
    expected_outcome = get_prediction_outcome(
        expected_parse_result,
        return_code=command_result.return_code,
        timed_out=command_result.timed_out,
    )

    actual_outcome, actual_parse_result, memory_used, runtime = postprocessor(
        command_result, {"1", "2"}
    )
    assert actual_outcome == expected_outcome
    assert actual_parse_result == expected_parse_result
    assert memory_used == 100
    assert math.isclose(runtime, 0.2)


def test_execution(basic_prediction, preprocessor, postprocessor, tmpdir):
    cwd = Path(tmpdir)
    imports, context, call_code = preprocessor(
        entry_point=basic_prediction["entry_point"],
        test_cases=basic_prediction["test_cases"],
        tc_timeout=0.25,
    )

    with open(cwd / "solution.py", "w") as f:
        f.write(basic_prediction["solution"])
    main_code = (
        imports
        + "\n"
        + context
        + "\n"
        + "benchmark_mode=False"
        + "\n"
        + call_code
    )
    with open(cwd / "main.py", "w") as f:
        f.write(main_code)

    cmd = ["python", "main.py"]

    result = serial_execute_code(
        CommandsToRun(
            cwd=cwd,
            commands=[Command(command=cmd, timeout=10, num_times=1)],
            tracked_files=[],
        )
    )
    assert result.last_cmd.return_code == 0
    assert not result.timed_out
    expected_stdout = [f"TEST_{i}___PASSED" for i in range(3)]
    expected_stdout.extend(["TEST_3___FAILED", "TEST_4___TypeError"])
    stdout = result.last_cmd.stdout.strip().split("\n")
    *test_results, mem, runtime_std = stdout
    assert test_results == expected_stdout
    assert mem.startswith("__MEMORY_USED__=")
    assert runtime_std.startswith("__RUNTIME__=")

    actual_outcome, _, memory_used, runtime = postprocessor(
        result.last_cmd, {"0", "1", "2", "3", "4"}
    )
    assert actual_outcome == PredictionOutcome.FAILED
    assert memory_used > 0
    assert runtime > 0


def test_make_test_does_not_remove_long():
    """Test that ensures that make test does not remove long test cases."""

    test_case = [
        "add_1()",
        repr([{"lat": 1.0, "lon": 1.0} for _ in range(10000)]),
        False,
    ]

    result = percent_passed._make_test_function(0, "add_1", test_case=test_case)
    assert len(result) > 0
