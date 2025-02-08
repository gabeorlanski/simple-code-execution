import logging
import re
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.data_structures import default_should_early_stop
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset import eval_utils
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig

logger = logging.getLogger(__name__)


# Adapted from https://github.com/Codium-ai/AlphaCodium/blob/main/alpha_codium/code_contests/data/prepare_and_clean_dataset.py#L152
def fix_test_problem_29(problem):
    generated_tests = problem.pop("generated_tests", {})

    new_generated_tests = {
        "input": [],
        "output": [],
    }
    for i, input in enumerate(generated_tests["input"]):
        is_valid = True
        for l in input.split():
            l_n = np.array(list(map(int, l.split())))
            if any(l_n < 0):  # according to the description, they should be >=0
                is_valid = False
                break

        s = input.split("\n", 1)
        n = int(s[0].strip())
        a = s[1].strip().split("\n")
        for j in range(n):
            num_elements = int(a[2 * j].strip())
            if num_elements != len(
                a[2 * j + 1].strip().split(" ")
            ):  # according to the description, they should be equal
                is_valid = False
                break
        if is_valid:
            new_generated_tests["input"].append(input)
            new_generated_tests["output"].append(generated_tests["output"][i])
    problem["generated_tests"] = new_generated_tests
    return problem


def fix_test_problem_92(problem: Dict):
    private_tests = problem.pop("private_tests", {})
    new_private_tests = {
        "input": [],
        "output": [],
    }
    for i, input in enumerate(private_tests["input"]):
        if (
            len(set(input)) == 4
        ):  # {'a', 'b',  '1', '\n'} - according to the description, the string should contain only 'a' and 'b'

            new_private_tests["input"].append(input)
            new_private_tests["output"].append(private_tests["output"][i])
    problem["private_tests"] = new_private_tests

    generated_tests = problem.pop("generated_tests", {})
    new_generated_tests = {
        "input": [],
        "output": [],
    }

    for i, input in enumerate(generated_tests["input"]):
        if (
            len(set(input)) == 4
        ):  # {'a', 'b',  '1', '\n'} - according to the description, the string should contain only 'a' and 'b'
            new_generated_tests["input"].append(input)
            new_generated_tests["output"].append(generated_tests["output"][i])
    problem["generated_tests"] = new_generated_tests
    return problem


def fix_problem_3_validation(problem):
    generated_tests = problem.pop("generated_tests", {})
    new_generated_tests = {
        "input": [],
        "output": [],
    }
    for i, input in enumerate(generated_tests["input"]):
        n, m, x = input.splitlines()[0].split()
        n = int(n)
        m = int(m)
        a = input.splitlines()[1].split()
        b = input.splitlines()[2].split()
        if n != len(a) or m != len(
            b
        ):  # according to the description, they should be equal
            pass
        else:
            new_generated_tests["input"].append(input)
            new_generated_tests["output"].append(generated_tests["output"][i])
    problem["generated_tests"] = new_generated_tests
    return problem


NAME_TO_FIXES = {
    "1582_B. Luntik and Subsequences": fix_test_problem_29,
    "1606_A. AB Balance": fix_test_problem_92,
    "1548_E. Gregor and the Two Painters": fix_problem_3_validation,
}


def _clean_stdout(stdout: str):
    # Based on https://github.com/google-deepmind/code_contests/blob/main/execution/tester_sandboxer.cc#L135
    stdout = stdout.strip().lower()
    stdout = re.sub(r"\n\t\r\v", "", stdout)
    return stdout


def filter_solutions(
    languages: List[int], solutions: List[str], keep_languages: Set[int]
) -> Tuple[List[int], List[str]]:

    langs = []
    sols = []
    for s, l in zip(solutions, languages):
        if l in keep_languages:
            langs.append(l)
            sols.append(s)

    return langs, sols


def process_problem(problem: Dict) -> Dict:
    if problem["name"] in NAME_TO_FIXES:
        problem = NAME_TO_FIXES[problem["name"]](problem)
    inputs = []
    outputs = []
    test_types = []
    for key, test_type in (
        ("public_tests", 0),
        ("private_tests", 1),
        ("generated_tests", 2),
    ):
        test_dict = problem.pop(key, {})
        for i, o in zip(test_dict["input"], test_dict["output"]):
            inputs.append(i)
            outputs.append(_clean_stdout(o))
            test_types.append(test_type)

    return {
        **problem,
        "inputs": inputs,
        "outputs": outputs,
        "test_types": test_types,
    }


def is_stdout_correct(actual: str, expected: str):

    if actual == expected:
        return True
    if actual.replace("", "") == expected.replace("", ""):
        return True

    try:
        actual = float(actual)
        expected = float(expected)
        return abs(actual - expected) < 1e-6
    except ValueError:
        return False


def should_stop_early(
    cmd_idx: int,
    res: CommandResult,
    expected_out: List[str],
    last_real_test_idx: Optional[int] = None,
    ensure_real_tests_run: bool = False,
) -> bool:
    """Determines if we should stop execution early."""

    # If we are ensuring that all REAL tests run, then we don't want to stop the
    # execution until we encounter a generated test.
    if (
        ensure_real_tests_run
        and last_real_test_idx is not None
        and cmd_idx < last_real_test_idx
    ):
        return False

    if res.had_error or res.timed_out:
        return True

    # The program is an assertion style, so if there is no error there is no
    # reason to stop early.
    if len(expected_out) <= cmd_idx:
        return True

    if not is_stdout_correct(
        _clean_stdout(res.stdout.replace("\r", "")), expected_out[cmd_idx]
    ):
        return True

    return False


def preprocess(
    problem: Dict,
    command_timeout: float,
    first_command_timeout: Optional[float],
    early_stopping: bool = False,
    disable_memory_limit: bool = False,
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    max_commands: Optional[int] = None,
    exclude_private: bool = False,
    exclude_generated: bool = False,
    ensure_real_tests_run: bool = False,
    python_command: str = "python3",
    force_command_timeout: bool = False,
) -> List[Executable]:

    inputs = problem["inputs"]
    if exclude_private or exclude_generated:
        new_inputs = []
        new_outputs = []
        new_test_types = []
        last_real_test_idx = None
        for i, t_type in enumerate(problem["test_types"]):
            if exclude_private and t_type == 1:
                continue
            if exclude_generated and t_type == 2:
                continue

            new_inputs.append(inputs[i])
            new_outputs.append(problem["outputs"][i])
            new_test_types.append(t_type)

            if t_type != 2:
                last_real_test_idx = len(new_inputs)
        inputs = new_inputs
        problem["outputs"] = new_outputs
        problem["test_types"] = new_test_types

    else:
        try:
            # Find the index of the first generated test.
            last_real_test_idx = problem["test_types"].index(2)
        except ValueError:
            # There are no generated tests.
            last_real_test_idx = None
    if not disable_memory_limit and problem["memory_limit_bytes"] > 0:
        mem_code = eval_utils.get_mem_limit_code(
            str(problem["memory_limit_bytes"]), "\n\n"
        )
    else:
        mem_code = ""

    if max_commands is not None:
        inputs = inputs[:max_commands]

    if problem.get("time_limit") is not None and not force_command_timeout:
        time_limit = problem["time_limit"]
        time_limit = time_limit["seconds"] + time_limit["nanos"] / 1e9
        time_limit = min(time_limit, command_timeout)
    else:
        time_limit = command_timeout

    if early_stopping:
        early_stop_fn = partial(
            should_stop_early,
            expected_out=problem["outputs"][: len(inputs)],
            ensure_real_tests_run=ensure_real_tests_run,
            last_real_test_idx=last_real_test_idx,
        )
    else:
        early_stop_fn = default_should_early_stop

    out = []
    for solution in problem[solution_list_key]:
        if isinstance(solution, str):
            program = solution
        else:
            program = solution[solution_str_key]

        program = mem_code + program

        executable = eval_utils.make_stdin_executable(
            files={"main.py": program},
            inputs=inputs,
            commands=[python_command, "main.py"],
            early_stop_fn=early_stop_fn,
            ensure_all_run=False,
            tracked_files=[],
            first_command_timeout=first_command_timeout or 0,
            command_timeout=time_limit,
            stdout_postprocess_fn=_clean_stdout,
        )
        out.append(executable)

    return out


def postprocess_program_result(
    pred: Dict | str,
    result: ExecutionResult,
    expected_outputs: List[str],
    test_types: List[int],
    num_stdout_save: int = None,
) -> bool:
    if isinstance(pred, str):
        pred = {"solution": pred}

    outcomes = []
    passed = (
        len(expected_outputs) == len(result.command_results)
        and result.all_had_return_code(0)
        and not result.timed_out
    )
    for i, expected in enumerate(expected_outputs):

        if i >= len(result):
            outcomes.append(False)
            passed = False
            continue

        res = result.command_results[i]
        if res.had_error or res.timed_out:
            outcomes.append(False)
            continue
        test_passed = is_stdout_correct(
            res.stdout,
            expected,
        )
        outcomes.append(test_passed)
        if not test_passed:
            passed = False

    # Check to see if we passed each test type.
    has_private = has_generated = False
    passed_public = passed_private = passed_generated = True
    for t_type, outcome in zip(test_types, outcomes):
        if t_type == 0:
            passed_public &= outcome
        elif t_type == 1:
            has_private = True
            passed_private &= outcome
        elif t_type == 2:
            has_generated = True
            passed_generated &= outcome

    stdout = [r.stdout for r in result.command_results]
    if num_stdout_save is not None:
        stdout = stdout[-num_stdout_save:]

    out_dict = {
        **pred,
        "timeout": result.timed_out,
        "had_error": result.had_error,
        "return_code": result.last_cmd.return_code,
        "passed": passed,
        "passed_public": passed_public,
        "outcomes": outcomes,
        "stderr": result.last_cmd.stderr,
        "command_elapsed": [r.runtime for r in result.command_results],
        "elapsed": result.elapsed,
        "stdout": stdout,
        "num_ran": len(result.command_results),
    }

    if has_generated:
        out_dict["passed_generated"] = passed_generated
    if has_private:
        out_dict["passed_private"] = passed_private
    return out_dict


def postprocess(
    problem: Dict,
    results: List[ExecutionResult],
    max_commands: Optional[int],
    solution_list_key: str = "solutions",
    exclude_private: bool = False,
    exclude_generated: bool = False,
    num_stdout_save: int = None,
):
    """Postprocesses the results of a list of predictions for a single problem.

    Args:
        problem (Dict): The problem to postprocess the results for.
        results (List[ExecutionResult]): The results of the predictions.
        max_commands (Optional[int]): The maximum number of commands to execute
            per problem.
        solution_list_key (str): The key for the list of solutions in the predictions.
        exclude_private (bool): Whether to exclude private tests.
        exclude_generated (bool): Whether to exclude generated tests.
        num_stdout_save (int): The number of stdout outputs to save for each
            prediction.
    Returns:
        A dictionary containing the postprocessed results.
    """
    expected_outputs = []
    expected_test_types = []
    for i, t_type in enumerate(problem["test_types"]):
        if max_commands is not None and len(expected_outputs) >= max_commands:
            break
        if exclude_private and t_type == 1:
            continue
        if exclude_generated and t_type == 2:
            continue
        expected_test_types.append(t_type)
        expected_outputs.append(problem["outputs"][i])

    out = []
    for i, res in enumerate(results):
        out.append(
            postprocess_program_result(
                problem[solution_list_key][i],
                res,
                expected_outputs,
                expected_test_types,
                num_stdout_save=num_stdout_save,
            )
        )

    return {
        **problem,
        "predictions": out,
    }


def evaluate(
    predictions: List[Dict],
    num_workers: int,
    first_command_timeout: int = 10,
    command_timeout: float = 1.0,
    max_commands: int = None,
    early_stopping: bool = False,
    disable_memory_limit: bool = False,
    k_vals: List[int] = None,
    execution_kwargs: Dict = None,
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    exclude_private: bool = False,
    exclude_generated: bool = False,
    ensure_real_tests_run: bool = False,
    python_command: str = "python3",
    force_command_timeout: bool = False,
    num_stdout_save: int = None,
) -> Tuple[Dict, List[Dict]]:
    """Evaluates predictions for CodeContests dataset.

    Assumes that predictions is a list of dictionaries where each element is a
    problem from the CodeContests dataset.

    Args:
        predictions: List[Dict]: A list of dictionaries where each element is a
            problem from the CodeContests dataset.
        num_workers (int): The number of workers to use for multiprocessing.
        first_command_timeout (int): The timeout for the first command in seconds.
        command_timeout (float): The timeout for each command in seconds. If
            there is a time_limit for the problem, then the min of the two will
            be used assuming force_command_timeout is False.
        max_commands (int): The maximum number of commands to execute per problem.
        early_stopping (bool): Whether to stop execution early if a test fails.
        disable_memory_limit (bool): Whether to disable memory limit set by each problem.
        k_vals (List[int]): A list of integers for pass@k evaluation.
        execution_kwargs (Dict): Additional keyword arguments to pass to the
            execution config.
        solution_str_key (str): The key for the solution string in the predictions.
        solution_list_key (str): The key for the list of solutions in the predictions.
        exclude_private (bool): Whether to exclude private tests.
        exclude_generated (bool): Whether to exclude generated tests.
        ensure_real_tests_run (bool): Whether to ensure that all non-generated tests are run.
        python_command (str): The command to use for executing Python code.
        force_command_timeout (bool): Whether to force the command timeout to be the
            maximum of the command_timeout and the time_limit set by each
            problem or just use command_timeout.
        num_stdout_save (int): The number of stdout outputs to save for each
            prediction.

    Returns:
        metrics (Dict): A dictionary containing the evaluation metrics.
        result (List[Dict]): A list of dictionaries containing the evaluation
            results for each problem.
    """
    logger.info("Evaluating predictions for code contests.")

    # If the predictions don't have inputs, then we need to process the problems.
    if "inputs" not in predictions[0]:
        logger.info("Processing problems.")
        predictions = [
            process_problem(p)
            for p in tqdm(predictions, desc="Processing problems")
        ]

    elapsed, results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(
            num_workers=num_workers, **(execution_kwargs or {})
        ),
        preprocessor=partial(
            preprocess,
            first_command_timeout=first_command_timeout,
            command_timeout=command_timeout,
            max_commands=max_commands,
            early_stopping=early_stopping,
            solution_str_key=solution_str_key,
            solution_list_key=solution_list_key,
            disable_memory_limit=disable_memory_limit,
            exclude_private=exclude_private,
            exclude_generated=exclude_generated,
            ensure_real_tests_run=ensure_real_tests_run,
            python_command=python_command,
            force_command_timeout=force_command_timeout,
        ),
        postprocessor=partial(
            postprocess,
            exclude_private=exclude_private,
            exclude_generated=exclude_generated,
            solution_list_key=solution_list_key,
            max_commands=max_commands,
            num_stdout_save=num_stdout_save,
        ),
        preproc_returns_list=True,
        return_elapsed=True,
    )

    num_samples = 1
    pass_counts = []
    for r in results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {
        "percent_passed": sum(pc > 0 for pc in pass_counts) / len(pass_counts),
        "elapsed": elapsed,
    }
    for k in k_vals or [1, 5, 10]:
        if k > num_samples:
            break
        metrics[f"pass@{k}"] = float(
            estimate_pass_at_k(
                [num_samples] * len(pass_counts), pass_counts, k
            ).mean()
        )

    return metrics, results
