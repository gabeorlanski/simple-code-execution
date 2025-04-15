import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from code_execution.data_structures import CommandResult
from code_execution.data_structures import ExecutionResult
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset.eval_utils import make_stdin_executable
from code_execution.eval_dataset.metrics import estimate_pass_at_k


class TestCodeContestsUtils:
    """Test utility functions for code contests evaluation."""

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
    def test_clean_stdout(self, input_str: str, expected: str):
        """Test cleaning of stdout output."""
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
    def test_is_stdout_correct(
        self, actual: str, expected: str, is_correct: bool
    ):
        """Test correctness checking of stdout output."""
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
        self,
        languages: List[int],
        solutions: List[str],
        keep_languages: Set[int],
        expected_langs: List[int],
        expected_sols: List[str],
    ):
        """Test filtering solutions by language."""
        filtered_langs, filtered_sols = code_contests.filter_solutions(
            languages, solutions, keep_languages
        )
        assert filtered_langs == expected_langs
        assert filtered_sols == expected_sols


class TestCodeContestsExecution:
    """Test execution-related functionality for code contests."""

    @pytest.fixture
    def sample_problem(self) -> Dict:
        """Sample problem fixture."""
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
    def sample_command_result(self) -> CommandResult:
        """Sample command result fixture."""
        return CommandResult(
            stdout="15\n",
            stderr="",
            return_code=0,
            timed_out=False,
            runtime=0.1,
        )

    @pytest.fixture
    def sample_execution_result(
        self, sample_command_result: CommandResult
    ) -> ExecutionResult:
        """Sample execution result fixture."""
        return ExecutionResult(
            command_results=[sample_command_result],
            elapsed=0.1,
            timed_out=False,
        )

    @pytest.fixture
    def multiple_solutions_problem(self) -> Dict:
        """Problem with multiple solutions fixture."""
        return {
            "name": "multiple_solutions",
            "memory_limit_bytes": 1024 * 1024 * 512,
            "time_limit": {"seconds": 10, "nanos": 5e6},
            "inputs": ["1 2 3", "4 5 6"],
            "outputs": ["6", "15"],
            "test_types": [0, 0],
            "solutions": [
                "print(sum(map(int, input().split())))",
                "nums = list(map(int, input().split()))\nprint(sum(nums))",
                "print(eval('+'.join(input().split())))",
            ],
        }

    @pytest.fixture
    def error_result(self) -> CommandResult:
        """Command result with error fixture."""
        return CommandResult(
            stdout="",
            stderr="error",
            return_code=1,
            timed_out=False,
            runtime=0.1,
        )

    @pytest.fixture
    def timeout_result(self) -> CommandResult:
        """Command result with timeout fixture."""
        return CommandResult(
            stdout="",
            stderr="",
            return_code=0,
            timed_out=True,
            runtime=0.1,
        )

    @pytest.fixture
    def success_result(self) -> CommandResult:
        """Command result with success fixture."""
        return CommandResult(
            stdout="15\n",
            stderr="",
            return_code=0,
            timed_out=False,
            runtime=0.1,
        )

    @pytest.fixture
    def wrong_output(self) -> CommandResult:
        """Command result with wrong output fixture."""
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
        self,
        request,
        test_idx: int,
        result_fixture: str,
        expected_outputs: List[str],
        last_real_test_idx: Optional[int],
        ensure_real_tests_run: bool,
        expected: bool,
    ):
        """Test early stopping logic for test execution."""
        result = request.getfixturevalue(result_fixture)
        actual = code_contests.should_stop_early(
            test_idx,
            result,
            expected_outputs,
            last_real_test_idx=last_real_test_idx,
            ensure_real_tests_run=ensure_real_tests_run,
        )
        assert actual == expected

    @pytest.mark.parametrize("force_command_timeout", [True, False])
    def test_preprocess(
        self,
        multiple_solutions_problem: Dict,
        force_command_timeout: bool,
    ):
        """Test preprocessing of problems into executables."""
        executables = code_contests.preprocess(
            problem=multiple_solutions_problem,
            command_timeout=2.0,
            first_command_timeout=30.0,
            disable_memory_limit=True,
            force_command_timeout=force_command_timeout,
        )

        assert len(executables) == 3

        for prog, exec in zip(
            multiple_solutions_problem["solutions"], executables
        ):
            expected = make_stdin_executable(
                files={"main.py": prog},
                inputs=multiple_solutions_problem["inputs"],
                commands=["python3", "main.py"],
                early_stop_fn=lambda **_: False,
                tracked_files=[],
                first_command_timeout=30.0,
                command_timeout=2.0 if force_command_timeout else 10.005,
                stdout_postprocess_fn=lambda s: s.strip(),
            )

            assert exec.files == expected.files
            assert exec.commands == expected.commands


class TestCodeContestsProblemFixes:
    """Test problem-specific fixes for code contests."""

    def test_fix_test_problem_29(self):
        """Test fix for problem 29's generated tests."""
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

    def test_fix_test_problem_92(self):
        """Test fix for problem 92's test cases."""
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

    def test_fix_problem_3_validation(self):
        """Test fix for problem 3's validation."""
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


class TestCodeContestsEvaluation:
    """Test evaluation functionality for code contests."""

    @pytest.fixture
    def real_problems(self) -> List[Dict]:
        """Load real problems from test data."""
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
        return data

    @pytest.mark.parametrize("should_pass", [True, False], ids=["pass", "fail"])
    @pytest.mark.parametrize(
        "exclude_generated", [True, False], ids=["no_generated", "generated"]
    )
    def test_evaluate(
        self,
        real_problems: List[Dict],
        should_pass: bool,
        exclude_generated: bool,
    ):
        """Test evaluation of code contest problems."""
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
                if k in {"solutions"}:
                    continue
                assert result[k] == v

            assert len(result["predictions"]) == len(expected_prob["solutions"])

            for j, expected_sol in enumerate(expected_prob["solutions"]):
                actual_sol = result["predictions"][j]
                assert actual_sol["language"] == expected_sol["language"]
                assert actual_sol["passed"] == should_pass
                assert "timing" in actual_sol
                assert set(actual_sol["timing"].keys()) == {
                    "writing",
                    "cleanup",
                    "cmd_exec",
                    "cmd_eval",
                    "preprocess",
                    "execution",
                    "postprocess",
                }


def make_execution_result(
    key: str,
    stdout: List[str],
    stderr: List[str] | str = "",
    return_code: List[int] | int = 0,
    runtime: List[float] | float = 0.1,
    timed_out: List[bool] | bool = False,
    writing_time: float = 0.03,
    cleanup_time: float = 0.02,
    preprocess_time: float = 0.04,
    expected_num_commands: Optional[int] = None,
    tracked_files: Optional[Dict[str, str]] = None,
    cwd: str = "/tmp",
):

    if not isinstance(stderr, list):
        stderr = [stderr] * len(stdout)
    if not isinstance(return_code, list):
        return_code = [return_code] * len(stdout)
    if not isinstance(runtime, list):
        runtime = [runtime] * len(stdout)
    if not isinstance(timed_out, list):
        timed_out = [timed_out] * len(stdout)
    if (
        len(stdout) != len(stderr)
        or len(stdout) != len(return_code)
        or len(stdout) != len(runtime)
        or len(stdout) != len(timed_out)
    ):
        raise ValueError("All arguments must be lists of the same length")

    net_elapsed = 0
    cmd_results = []
    for out, err, r_code, r_time, to in zip(
        stdout, stderr, return_code, runtime, timed_out
    ):
        cmd_results.append(
            CommandResult(
                return_code=r_code,
                runtime=r_time,
                stdout=out,
                stderr=err,
                timed_out=to,
            )
        )
        net_elapsed += r_time
    return ExecutionResult(
        key=key,
        command_results=cmd_results,
        elapsed=net_elapsed,
        cwd=cwd,
        tracked_files=tracked_files or {},
        expected_num_commands=expected_num_commands or len(cmd_results),
        writing_time=writing_time,
        cleanup_time=cleanup_time,
        preprocess_time=preprocess_time,
    )


class TestCodeContestsPostprocessing:
    """Test postprocessing functionality for code contests."""

    @pytest.fixture
    def expected_outputs(self) -> List[str]:
        """Expected outputs fixture."""
        return [
            "Hello",
            "He ll o",
            "He  ll  o",
            "1.23",
        ]

    @pytest.fixture
    def successful_result(
        self, request, expected_outputs: List[str]
    ) -> ExecutionResult:

        if not hasattr(request, "param"):
            request.param = {}
        """Successful execution result fixture."""
        return make_execution_result(
            key="test1",
            stdout=[
                o
                for _, o in zip(
                    range(
                        request.param.get("num_stdout", len(expected_outputs))
                    ),
                    expected_outputs,
                )
            ],
        )

    @pytest.fixture
    def failure_result(
        self, request, expected_outputs: List[str]
    ) -> ExecutionResult:
        """Failure execution result fixture."""
        expected_offset = [expected_outputs[0]] + ["ERROR"] * (
            len(expected_outputs) - 1
        )
        if not hasattr(request, "param"):
            request.param = {}

        return make_execution_result(
            key="test1",
            stdout=[
                o
                for _, o in zip(
                    range(request.param.get("num_stdout", 1)), expected_offset
                )
            ],
            expected_num_commands=request.param.get(
                "expected_num_commands", len(expected_outputs)
            ),
        )

    @pytest.mark.parametrize(
        "successful_result",
        [
            {"num_stdout": 1},
            {"num_stdout": 10},
        ],
        ids=["single", "multiple"],
        indirect=True,
    )
    def test_postprocess_result(
        self,
        successful_result: ExecutionResult,
        expected_outputs: List[str],
    ):
        """Test postprocessing of program results."""
        expected_stdout = [
            o
            for _, o in zip(successful_result.command_results, expected_outputs)
        ]
        actual = code_contests.postprocess_program_result(
            {"solution": "test"},
            successful_result,
            expected_outputs=expected_stdout,
            test_types=[0] * len(expected_outputs),
        )

        assert actual["solution"] == "test"
        assert actual["passed"]
        assert "passed_generated" not in actual
        assert "passed_private" not in actual

        assert actual["outcomes"] == [True] * len(expected_stdout)
        assert len(actual["timing"]["cmd_eval"]) == len(expected_stdout)
        assert actual["timing"]["cmd_exec"] == [0.1] * len(expected_stdout)
        assert actual["stdout"] == expected_stdout

        assert actual["stderr"] == ""
        assert not actual["timeout"]
        assert not actual["had_error"]
        assert actual["return_code"] == 0
        assert actual["timing"]["writing"] == successful_result.writing_time
        assert actual["timing"]["cleanup"] == successful_result.cleanup_time
        assert (
            actual["timing"]["preprocess"] == successful_result.preprocess_time
        )
        assert actual["timing"]["execution"] == successful_result.elapsed
        assert actual["num_ran"] == len(expected_stdout)

    @pytest.mark.parametrize(
        "test_types, expected_passed",
        [
            ([0, 1, 2], (True, True, False)),
            ([0, 1, 1], (True, False, None)),
            ([0, 0, 1], (True, False, None)),
            ([0, 2, 2], (True, None, False)),
            ([0, 0, 2], (True, None, False)),
            ([1, 1, 1], (None, False, None)),
            ([2, 2, 2], (None, None, False)),
            ([1, 1, 0], (False, True, None)),
            ([2, 2, 0], (False, None, True)),
            ([2, 2, 1], (None, False, True)),
            ([1, 1, 2], (None, True, False)),
            ([0, 0, 0], (False, None, None)),
        ],
        ids=[
            "pub_pri_gen",
            "pub_pri_pri",
            "pub_pub_pri",
            "pub_gen_gen",
            "pub_pub_gen",
            "pri_pri_pri",
            "gen_gen_gen",
            "pri_pri_pub",
            "gen_gen_pub",
            "gen_gen_pri",
            "pri_pri_gen",
            "pub_pub_pub",
        ],
    )
    def test_postprocess_result_test_types_fail(
        self,
        expected_outputs: List[str],
        test_types: List[int],
        expected_passed: Tuple[bool, bool, bool],
    ):
        """Test postprocessing of program results."""
        result = make_execution_result(
            key="test1",
            stdout=[expected_outputs[0], expected_outputs[1], "c"],
        )
        actual = code_contests.postprocess_program_result(
            {"solution": "test"},
            result,
            expected_outputs=expected_outputs[:3],
            test_types=test_types,
        )

        assert not actual["passed"]

        assert actual.get("passed_public", None) == expected_passed[0]
        assert actual.get("passed_private", None) == expected_passed[1]
        assert actual.get("passed_generated", None) == expected_passed[2]

    @pytest.mark.parametrize(
        "test_types",
        [
            [0] * 2 + [1] * 8,
            [0] * 2 + [2] * 8,
            [0] * 2 + [1] * 4 + [2] * 4,
        ],
        ids=["passed_private", "passed_generated", "passed_both"],
    )
    def test_postprocess_fail_test_types(
        self,
        successful_result: ExecutionResult,
        expected_outputs: List[str],
        test_types: List[int],
    ):
        """Test postprocessing of program results."""
        actual = code_contests.postprocess_program_result(
            {"solution": "test"},
            successful_result,
            expected_outputs=expected_outputs,
            test_types=test_types,
        )

        assert actual["solution"] == "test"
        assert actual["passed"]

        has_generated = 2 in test_types
        has_private = 1 in test_types
        if has_generated:
            assert actual["passed_generated"]
        else:
            assert "passed_generated" not in actual
        if has_private:
            assert actual["passed_private"]
        else:
            assert "passed_private" not in actual

    @pytest.mark.parametrize(
        "failure_result",
        [
            {"num_stdout": 5, "expected_num_commands": 5},
            {"num_stdout": 1, "expected_num_commands": 5},
        ],
        ids=["all_present", "some_missing"],
        indirect=True,
    )
    def test_postprocess_result_failure(
        self,
        failure_result: ExecutionResult,
        expected_outputs: List[str],
    ):
        """Test postprocessing of program results."""
        expected_stdout = [o for _, o in zip(range(5), expected_outputs)]
        actual = code_contests.postprocess_program_result(
            {"solution": "test"},
            failure_result,
            expected_outputs=expected_stdout,
            test_types=[0] * len(expected_outputs),
        )

        assert actual["solution"] == "test"
        assert not actual["passed"]
        assert "passed_generated" not in actual
        assert "passed_private" not in actual

        expected_outcomes = [True] + [False] * (
            len(failure_result.command_results) - 1
        )

        assert actual["outcomes"] == expected_outcomes
        assert actual["num_ran"] == len(failure_result.command_results)
        assert len(actual["timing"]["cmd_eval"]) == len(
            failure_result.command_results
        )
        assert actual["timing"]["cmd_exec"] == [0.1] * len(
            failure_result.command_results
        )
        assert actual["stdout"] == [
            c.stdout for c in failure_result.command_results
        ]

        assert actual["stderr"] == ""
        assert not actual["timeout"]
        assert not actual["had_error"]
        assert actual["return_code"] == 0
        assert actual["timing"]["writing"] == failure_result.writing_time
        assert actual["timing"]["cleanup"] == failure_result.cleanup_time
        assert actual["timing"]["preprocess"] == failure_result.preprocess_time
        assert actual["timing"]["execution"] == failure_result.elapsed
        assert actual["num_ran"] == len(failure_result.command_results)
