import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from code_execution.data_structures import CommandResult
from code_execution.data_structures import ExecutionResult
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset.eval_utils import make_stdin_executable
from code_execution.eval_dataset.metrics import estimate_pass_at_k


# Mock data and fixtures
@pytest.fixture
def sample_problem():
    return {
        "name": "test_problem",
        "memory_limit_bytes": 1024 * 1024 * 512,  # 512MB
        "time_limit": {"seconds": 1, "nanos": 0},
        "inputs": ["5\n1 2 3 4 5", "3\n6 7 8", "2\n9 10"],
        "outputs": ["15", "21", "19"],
        "test_types": [0, 1, 2],  # public, private, and generated test
        "solutions": ["print(sum(map(int, input().split()[1:])))"],
    }


@pytest.fixture
def sample_command_result():
    return CommandResult(
        stdout="15\n",
        stderr="",
        return_code=0,
        timed_out=False,
        runtime=0.1,
    )


@pytest.fixture
def sample_execution_result(sample_command_result):
    return ExecutionResult(
        command_results=[sample_command_result],
        elapsed=0.1,
        timed_out=False,
    )


@pytest.fixture
def multiple_solutions_problem():
    return {
        "name": "multiple_solutions",
        "memory_limit_bytes": 1024 * 1024 * 512,
        "time_limit": {"seconds": 1, "nanos": 0},
        "inputs": ["1 2 3", "4 5 6"],
        "outputs": ["6", "15"],
        "test_types": [0, 0],
        "solutions": [
            "print(sum(map(int, input().split())))",
            "nums = list(map(int, input().split()))\nprint(sum(nums))",
            "print(eval('+'.join(input().split())))",
        ],
    }


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("  15\n\t\r\v  ", "15"),
        ("TEST\nOUTPUT", "test\noutput"),
        ("", ""),
        ("1.000\n", "1.000"),
        ("YES\n\n\n", "yes"),
        ("\t\t\tHello\t\t\t", "hello"),
    ],
)
def test_clean_stdout(input_str, expected):
    assert code_contests._clean_stdout(input_str) == expected


@pytest.mark.parametrize(
    "actual, expected, is_correct",
    [
        ("15", "15", True),
        ("15  ", "15", True),
        ("  15", "15", True),
        ("1.000001", "1.0", True),
        ("2.0", "1.0", False),
        ("0.999999999999", "1.0", True),
        ("0.999999", "1.0", False),
        ("hello", "hello", True),
        ("hello", "world", False),
        ("", "", True),
        (" ", "", True),
        ("1e-6", "0.000001", True),
    ],
)
def test_is_stdout_correct(actual, expected, is_correct):
    assert code_contests.is_stdout_correct(actual, expected) == is_correct


@pytest.mark.parametrize(
    "languages, solutions, keep_languages, expected_langs, expected_sols",
    [
        (
            [1, 2, 3, 4, 5],
            ["sol1", "sol2", "sol3", "sol4", "sol5"],
            {1},
            [1],
            ["sol1"],
        ),
        (
            [1, 2, 3, 4, 5],
            ["sol1", "sol2", "sol3", "sol4", "sol5"],
            {1, 3, 5},
            [1, 3, 5],
            ["sol1", "sol3", "sol5"],
        ),
        (
            [1, 2, 3, 4, 5],
            ["sol1", "sol2", "sol3", "sol4", "sol5"],
            set(),
            [],
            [],
        ),
    ],
    ids=["single language", "multiple languages", "empty keep_languages"],
)
def test_filter_solutions(
    languages, solutions, keep_languages, expected_langs, expected_sols
):
    filtered_langs, filtered_sols = code_contests.filter_solutions(
        languages, solutions, keep_languages
    )
    assert filtered_langs == expected_langs
    assert filtered_sols == expected_sols


@pytest.fixture
def error_result():
    return CommandResult(
        stdout="",
        stderr="error",
        return_code=1,
        timed_out=False,
        runtime=0.1,
    )


@pytest.fixture
def timeout_result():
    return CommandResult(
        stdout="",
        stderr="",
        return_code=0,
        timed_out=True,
        runtime=0.1,
    )


@pytest.fixture
def success_result():
    return CommandResult(
        stdout="15\n",
        stderr="",
        return_code=0,
        timed_out=False,
        runtime=0.1,
    )


@pytest.fixture
def wrong_output():
    return CommandResult(
        stdout="16\n",
        stderr="",
        return_code=0,
        timed_out=False,
        runtime=0.1,
    )


@pytest.mark.parametrize(
    "test_idx, result_fixture, expected_outputs, last_real_test_idx, ensure_real_tests_run, expected",
    [
        # Basic cases
        (0, "error_result", ["15"], None, False, True),
        (0, "timeout_result", ["15"], None, False, True),
        (0, "success_result", ["15"], None, False, False),
        # Cases with ensure_real_tests_run
        (0, "error_result", ["15"], 1, True, False),
        (1, "error_result", ["15", "16"], 1, True, True),
        # Cases with wrong output but must run real tests
        (0, "wrong_output", ["15"], 1, True, False),
        (2, "wrong_output", ["15"], 1, True, True),
    ],
)
def test_should_stop_early_comprehensive(
    request,
    test_idx,
    result_fixture,
    expected_outputs,
    last_real_test_idx,
    ensure_real_tests_run,
    expected,
):
    # Get the fixture result
    result = request.getfixturevalue(result_fixture)

    # Call the function with the appropriate parameters
    actual = code_contests.should_stop_early(
        test_idx,
        result,
        expected_outputs,
        last_real_test_idx=last_real_test_idx,
        ensure_real_tests_run=ensure_real_tests_run,
    )

    # Assert the result
    assert actual == expected


def test_preprocess(
    multiple_solutions_problem,
):
    executables = code_contests.preprocess(
        problem=multiple_solutions_problem,
        command_timeout=1.0,
        first_command_timeout=2.0,
        disable_memory_limit=True,
    )

    assert len(executables) == 3

    for prog, exec in zip(multiple_solutions_problem["solutions"], executables):
        expected = make_stdin_executable(
            files={"main.py": prog},
            inputs=multiple_solutions_problem["inputs"],
            commands=["python3", "main.py"],
            early_stop_fn=lambda **_: False,
            ensure_all_run=False,
            tracked_files=[],
            first_command_timeout=2.0,
            command_timeout=1.0,
            stdout_postprocess_fn=lambda s: s.strip(),
        )

        assert exec.files == expected.files
        assert exec.commands == expected.commands


def test_fix_test_problem_29():
    # Test code_contests.fix_test_problem_29
    problem_29 = {
        "name": "1582_B. Luntik and Subsequences",
        "generated_tests": {
            "input": [
                "2\n1\n-1\n1\n1",  # Invalid negative number
                "2\n1\n1\n2\n1 2",  # Valid test
                "2\n2\n123\n1\n1333",  # Invalid length mismatch
            ],
            "output": ["YES", "NO", "YES"],
        },
    }
    fixed_29 = code_contests.fix_test_problem_29(copy.deepcopy(problem_29))
    assert fixed_29["generated_tests"]["input"] == [
        problem_29["generated_tests"]["input"][1],
    ]
    assert fixed_29["generated_tests"]["output"] == [
        problem_29["generated_tests"]["output"][1],
    ]


def test_fix_test_problem_92():
    # Test code_contests.fix_test_problem_92
    problem_92 = {
        "name": "1606_A. AB Balance",
        "private_tests": {
            "input": [
                "1\nab",  # Valid
                "abc\n",  # Invalid character
                "a\n1b",  # Invalid character
                "1\nba\nbbbbaaa",  # Valid
            ],
            "output": ["YES", "NO", "NO", "YES"],
        },
        "generated_tests": {
            "input": [
                "a2b",  # Invalid character
                "5\nab",  # Valid
                "abc",  # Invalid character
            ],
            "output": ["YES", "NO", "NO"],
        },
        "unchanged": {
            "input": [],
            "output": [],
        },
    }
    fixed_92 = code_contests.fix_test_problem_92(copy.deepcopy(problem_92))
    assert fixed_92["private_tests"]["input"] == [
        problem_92["private_tests"]["input"][0],
        problem_92["private_tests"]["input"][3],
    ]
    assert fixed_92["private_tests"]["output"] == [
        problem_92["private_tests"]["output"][0],
        problem_92["private_tests"]["output"][3],
    ]

    assert fixed_92["generated_tests"]["input"] == [
        problem_92["generated_tests"]["input"][1],
    ]
    assert fixed_92["generated_tests"]["output"] == [
        problem_92["generated_tests"]["output"][1],
    ]
    assert fixed_92["unchanged"] == problem_92["unchanged"]


def test_fix_problem_3_validation():
    # Test code_contests.fix_problem_3_validation
    problem_3 = {
        "name": "1548_E. Gregor and the Two Painters",
        "generated_tests": {
            "input": [
                "2 2 1\n1 2\n3 4",  # Valid
                "2 3 1\n1 2\n3 4 5",  # Valid
                "2 2 1\n1\n3 4",  # Invalid length mismatch
            ],
            "output": ["YES", "NO", "YES"],
        },
    }
    fixed_3 = code_contests.fix_problem_3_validation(problem_3)
    assert len(fixed_3["generated_tests"]["input"]) == 2
    assert len(fixed_3["generated_tests"]["output"]) == 2


@pytest.fixture
def real_problems():
    data_path = Path(__file__).parent / "code_contests_data.json"
    with data_path.open() as f:
        data = []
        for problem in json.load(f):
            for k in ["solutions", "incorrect_solutions"]:
                problem[k] = [
                    {"language": l, "solution": s}
                    for l, s in zip(
                        problem[k]["language"], problem[k]["solution"]
                    )
                ]
            data.append(problem)
    yield data


@pytest.mark.parametrize("should_pass", [True, False], ids=["pass", "fail"])
@pytest.mark.parametrize(
    "exclude_generated", [True, False], ids=["no_generated", "generated"]
)
def test_evaluate(real_problems, should_pass, exclude_generated):
    problems = []
    for problem in real_problems:

        if not should_pass:
            problem.pop("solutions")
            problem["solutions"] = problem.pop("incorrect_solutions")
        else:
            problem.pop("incorrect_solutions")
        problems.append(problem)

    metrics, preds = code_contests.evaluate(
        problems,
        1,
        exclude_generated=exclude_generated,
        force_command_timeout=True,
    )
    assert set(metrics.keys()) == {
        "pass@1",
        "percent_passed",
        "net_time",
        "pure_exec_time",
        "execution_time",
        "writing_time",
        "postprocessing_time",
        "preprocessing_time",
        "timestamp",
    }

    expected = 1.0 if should_pass else 0.0

    expected_pass_at_k = estimate_pass_at_k(
        [3, 3, 3], [1, 3, 3] if should_pass else [0, 0, 0], 1
    ).mean()

    assert abs(metrics["pass@1"] - expected_pass_at_k) < 1e-6
    assert abs(metrics["percent_passed"] - expected) < 1e-6

    assert len(preds) == len(problems)
    for i, expected_prob in enumerate(problems):
        result = preds[i]
        for k, v in expected_prob.items():
            if k in {
                "solutions",
            }:
                continue
            assert result[k] == v

        assert len(result["predictions"]) == len(expected_prob["solutions"])

        for j, expected_sol in enumerate(expected_prob["solutions"]):
            actual_sol = result["predictions"][j]
            assert actual_sol["language"] == expected_sol["language"]
            assert actual_sol["passed"] == should_pass
            assert "postprocess_time" in actual_sol
            assert "preprocess_time" in actual_sol
            assert "writing_time" in actual_sol
            assert "cleanup_time" in actual_sol
