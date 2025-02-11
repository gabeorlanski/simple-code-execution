import pytest
import numpy as np
from typing import Dict, List
from code_execution.data_structures import CommandResult, ExecutionResult
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset.eval_utils import make_stdin_executable


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


@pytest.mark.parametrize(
    "problem, expected",
    [
        ({"test_types": [0, 1, 2]}, {"test_types": [0, 1, 2]}),
        ({"test_types": [0, 1]}, {"test_types": [0, 1]}),
        ({"test_types": [0]}, {"test_types": [0]}),
        ({"test_types": [1]}, {"test_types": [1]}),
        ({"test_types": [2]}, {"test_types": [2]}),
        ({}, {"test_types": [0, 1, 2]}),
    ],
    ids=[
        "all test types",
        "public and private",
        "only public",
        "only private",
        "only generated",
        "no test types provided",
    ],
)
def test_process_problem_with_all_test_types(sample_problem, problem, expected):
    processed = code_contests.process_problem({**sample_problem, **problem})

    assert "test_types" in processed
    assert processed["test_types"] == expected["test_types"]


def test_should_stop_early_comprehensive():
    # Base cases
    error_result = CommandResult(
        stdout="",
        stderr="error",
        return_code=1,
        timed_out=False,
        runtime=0.1,
    )
    timeout_result = CommandResult(
        stdout="",
        stderr="",
        return_code=0,
        timed_out=True,
        runtime=0.1,
    )
    success_result = CommandResult(
        stdout="15\n",
        stderr="",
        return_code=0,
        timed_out=False,
        runtime=0.1,
    )

    # Test basic cases
    assert code_contests.should_stop_early(0, error_result, ["15"]) == True
    assert code_contests.should_stop_early(0, timeout_result, ["15"]) == True
    assert code_contests.should_stop_early(0, success_result, ["15"]) == False

    # Test with ensure_real_tests_run
    assert (
        code_contests.should_stop_early(
            0,
            error_result,
            ["15"],
            last_real_test_idx=1,
            ensure_real_tests_run=True,
        )
        == False
    )
    assert (
        code_contests.should_stop_early(
            1,
            error_result,
            ["15", "16"],
            last_real_test_idx=1,
            ensure_real_tests_run=True,
        )
        == True
    )

    # Test with wrong output but must run real tests
    wrong_output = CommandResult(
        stdout="16\n",
        stderr="",
        return_code=0,
        timed_out=False,
        runtime=0.1,
    )
    assert (
        code_contests.should_stop_early(
            0,
            wrong_output,
            ["15"],
            last_real_test_idx=1,
            ensure_real_tests_run=True,
        )
        == False
    )
    assert (
        code_contests.should_stop_early(
            2,
            wrong_output,
            ["15"],
            last_real_test_idx=1,
            ensure_real_tests_run=True,
        )
        == True
    )


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


def test_fix_test_problems():
    # Test code_contests.fix_test_problem_29
    problem_29 = {
        "name": "1582_B. Luntik and Subsequences",
        "generated_tests": {
            "input": [
                "2\n1\n1\n2\n1 2",  # Valid test
                "2\n1\n-1\n1\n1",  # Invalid negative number
                "2\n2\n1 2\n1\n1",  # Invalid length mismatch
            ],
            "output": ["YES", "NO", "YES"],
        },
    }
    fixed_29 = code_contests.fix_test_problem_29(problem_29)
    assert len(fixed_29["generated_tests"]["input"]) == 1
    assert len(fixed_29["generated_tests"]["output"]) == 1

    # Test code_contests.fix_test_problem_92
    problem_92 = {
        "name": "1606_A. AB Balance",
        "private_tests": {
            "input": [
                "aba",  # Valid
                "abc",  # Invalid character
                "a1b",  # Invalid character
                "abba",  # Valid
            ],
            "output": ["YES", "NO", "NO", "YES"],
        },
        "generated_tests": {
            "input": [
                "ab",  # Valid
                "a2b",  # Invalid character
                "abc",  # Invalid character
            ],
            "output": ["YES", "NO", "NO"],
        },
    }
    fixed_92 = code_contests.fix_test_problem_92(problem_92)
    assert len(fixed_92["private_tests"]["input"]) == 2
    assert len(fixed_92["private_tests"]["output"]) == 2
    assert len(fixed_92["generated_tests"]["input"]) == 1
    assert len(fixed_92["generated_tests"]["output"]) == 1

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


def test_postprocess_program_result_comprehensive():
    # Test successful case
    success_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="15\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
            CommandResult(
                stdout="21\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
        ],
        elapsed=0.2,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=success_result,
        expected_outputs=["15", "21"],
        test_types=[0, 1],
    )
    assert result["passed"] == True
    assert result["passed_public"] == True
    assert result["passed_private"] == True
    assert len(result["outcomes"]) == 2
    assert all(result["outcomes"])

    # Test partial success
    partial_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="15\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
            CommandResult(
                stdout="20\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
        ],
        elapsed=0.2,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=partial_result,
        expected_outputs=["15", "21"],
        test_types=[0, 1],
    )
    assert result["passed"] == False
    assert result["passed_public"] == True
    assert result["passed_private"] == False

    # Test error case
    error_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="",
                stderr="error",
                return_code=1,
                timed_out=False,
                runtime=0.1,
            )
        ],
        elapsed=0.1,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=error_result,
        expected_outputs=["15"],
        test_types=[0],
    )
    assert result["passed"] == False
    assert result["had_error"] == True
    assert len(result["outcomes"]) == 1
    assert not any(result["outcomes"])

    # Test timeout case
    timeout_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="",
                stderr="",
                return_code=0,
                timed_out=True,
                runtime=2.0,
            )
        ],
        elapsed=2.0,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=timeout_result,
        expected_outputs=["15"],
        test_types=[0],
    )
    assert result["passed"] == False
    assert result["timeout"] == True
    assert len(result["outcomes"]) == 1
    assert not any(result["outcomes"])

    # Test mixed results with all test types
    mixed_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="15\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
            CommandResult(
                stdout="21\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
            CommandResult(
                stdout="wrong\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            ),
        ],
        elapsed=0.3,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=mixed_result,
        expected_outputs=["15", "21", "19"],
        test_types=[0, 1, 2],  # public, private, generated
    )
    assert result["passed"] == False
    assert result["passed_public"] == True
    assert result["passed_private"] == True
    assert result["passed_generated"] == False
    assert len(result["outcomes"]) == 3
    assert result["outcomes"] == [True, True, False]

    # Test incomplete execution (fewer results than expected outputs)
    incomplete_result = ExecutionResult(
        command_results=[
            CommandResult(
                stdout="15\n",
                stderr="",
                return_code=0,
                timed_out=False,
                runtime=0.1,
            )
        ],
        elapsed=0.1,
    )

    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=incomplete_result,
        expected_outputs=["15", "21"],
        test_types=[0, 1],
    )
    assert result["passed"] == False
    assert len(result["outcomes"]) == 2
    assert result["outcomes"] == [True, False]

    # Test with stdout saving limit
    result = code_contests.postprocess_program_result(
        pred="test_solution",
        result=success_result,
        expected_outputs=["15", "21"],
        test_types=[0, 1],
        num_stdout_save=1,
    )
    assert len(result["stdout"]) == 1

    # Test with dictionary prediction instead of string
    result = code_contests.postprocess_program_result(
        pred={"solution": "test_solution", "metadata": "test"},
        result=success_result,
        expected_outputs=["15", "21"],
        test_types=[0, 1],
    )
    assert "metadata" in result
    assert result["metadata"] == "test"
