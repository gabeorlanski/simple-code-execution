""" Module for evaluating apps dataset. """

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import ujson

from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.data_structures import default_should_early_stop
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset import eval_utils
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig
from code_execution.utils import get_mem_limit_code

logger = logging.getLogger(__name__)
NO_FN_NAME = "__NO_FN_NAME__"
APPS_IMPORT_STR = """import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import heappush, heappop, heapify, heappushpop, heapreplace, merge, nlargest, nsmallest"""

APPS_FIXES = {
    4120: {
        "inputs": [
            [{1: 3, 2: 2, 3: 1}],
            [{1: 2, 2: 4, 3: 6}],
            [{1: 5, 3: 10, 2: 2, 6: 3, 8: 8}],
            [{"a": 6, "b": 2, "c": 4}],
        ]
    }
}

APPS_OUTPUT_CONVERTER = """def convert_output(output):
    if isinstance(output,tuple):
        output= list(output)
    try:
        if isinstance(output[0], tuple):
            output = [list(o) for o in output]
    except:
        pass
    return output"""

APPS_ASSERT_IF_TEMPLATE = """import argparse
_ARG_PARSER_ = argparse.ArgumentParser()
_ARG_PARSER_.add_argument('test_id', type=int)
__ARGS__= _ARG_PARSER_.parse_args()

%%%TEST_IFS%%%
else:
    raise ValueError(f"Unknown test_id: {__ARGS__.test_id}")
print(f"Passed test: {__ARGS__.test_id}")"""


APPS_IF_BRANCH = """if __ARGS__.test_id == %%%test_id%%%:
    %%%test_case%%%"""


def make_test_case(fn_name, inputs, outputs):
    if isinstance(outputs, list) and len(outputs) == 1:
        expected_out = repr(outputs[0])
    else:
        expected_out = repr(outputs)
    return f"assert convert_output({fn_name}({','.join(map(repr,inputs))})) == {expected_out}"


def process_raw_example(
    example, solution_col: str = "solutions", asserts_as_list: bool = True
):
    """Processes a raw example from the APPs dataset.

    Converts the input_output strings and solutions string to lists of inputs and outputs.

    For asserts style problems, it will also determine if the solution is a class or not. It will also populate the inputs field with the formatted test cases.

    """
    try:
        input_output = ujson.loads(example["input_output"])
    except (ujson.JSONDecodeError, ValueError):
        input_output = {"inputs": [], "outputs": []}

    try:
        solutions = ujson.loads(example[solution_col])
    except ujson.JSONDecodeError:
        solutions = []

    # check if the solution is a class, so we can call the function Solution.fn_name

    use_fn_name = NO_FN_NAME
    if "fn_name" in input_output:

        is_cls = sum("class Solution" in s for s in solutions)
        if is_cls / len(solutions) > 0.5:

            use_fn_name = "Solution()." + input_output["fn_name"]
        else:
            use_fn_name = input_output["fn_name"]
    # convert these to strings for pyarrow
    new_inputs = []
    new_outputs = []

    for inp, out in zip(input_output["inputs"], input_output["outputs"]):

        if use_fn_name != NO_FN_NAME:
            test_case = make_test_case(use_fn_name, inp, out)

            # This is because pyarrow datasets will throw a fit.
            if asserts_as_list:
                new_inputs.append([test_case])
            else:
                new_inputs.append(test_case)
        elif isinstance(inp, list):
            new_inputs.append(inp)
            new_outputs.append("\n".join(out))
        else:
            new_inputs.append(inp.split("\n"))
            new_outputs.append(out)

    return {
        **example,
        "inputs": new_inputs,
        "outputs": new_outputs,
        "solutions": solutions,
        "exec_mode": "asserts" if use_fn_name != NO_FN_NAME else "stdin",
        "fn_name": use_fn_name,
    }


def should_stop_early(
    cmd_idx: int, res: CommandResult, is_stdin: bool, expected_out: List[str]
) -> bool:
    """Determines if we should stop execution early."""

    if res.had_error or res.timed_out:
        return True

    # The program is an assertion style, so if there is no error there is no
    # reason to stop early.
    if not is_stdin:
        return False

    if len(expected_out) <= cmd_idx:
        return True

    if res.stdout.replace("\r", "") != expected_out[cmd_idx]:
        return True

    return False


def make_executable(
    solution: str | Dict,
    inputs: List[List[str]],
    outputs: List[str],
    exec_mode: str,
    first_command_timeout: Optional[float] = None,
    command_timeout: float = 2.0,
    max_commands: int = None,
    early_stopping: bool = False,
    solution_str_key: str = "solution",
    max_memory: str = "4*1024*1024*1024",
) -> Executable:
    """Makes the executable for an APPS problem.

    If the program is an assertion based program, it will add each test case as
    an if statement and expects a command line argument of the test idx to run.

    Args:
        solution (str | Dict): The solution code.
        inputs (List[List[str]]): The inputs to the program.
        outputs (List[str]): The expected outputs if the problem is stdin based.
        exec_mode (str): The type of execution for the program. Either "stdin" or "asserts".
        first_command_timeout (Optional[float], optional): _description_. Defaults to None.
        command_timeout (float, optional): The individual command timeout. Defaults to 2.0.
        max_commands (int, optional): The maximum commands to run. Defaults to None.
        early_stopping (bool, optional): Stop on the first incorrect output. Defaults to False.
        solution_str_key (str, optional): The key in the solution dict where the raw code is. Defaults to "solution".
        max_memory (str, optional): Max memory to use. Defaults to "4*1024*1024*1024".

    Returns:
        Executable: The executable for the problem + solution.
    """
    if isinstance(solution, dict):
        solution_str = solution[solution_str_key]
    else:
        solution_str = solution

    if exec_mode == "asserts":
        solution_str = (
            APPS_IMPORT_STR
            + "\n\n"
            + APPS_OUTPUT_CONVERTER
            + "\n\n"
            + solution_str
        )
    if early_stopping:
        early_stop_fn = partial(
            should_stop_early,
            expected_out=outputs,
            is_stdin=exec_mode == "stdin",
        )
    else:
        early_stop_fn = default_should_early_stop
    solution_str = (
        get_mem_limit_code(max_memory, trailing="\n\n") + solution_str
    )
    files = {"main.py": solution_str}
    if exec_mode == "stdin":
        if max_commands:
            inputs = inputs[:max_commands]

        return eval_utils.make_stdin_executable(
            files=files,
            inputs=inputs,
            commands=["python3", "main.py"],
            early_stop_fn=early_stop_fn,
            first_command_timeout=first_command_timeout,
            command_timeout=command_timeout,
        )

    # If it reaches this point it must be asserts.
    if exec_mode != "asserts":
        raise ValueError(f"Unknown exec_mode: {exec_mode}")

    # Leetcode needs imports to be in the solution
    test_cases = []
    commands = []
    for tc_l in inputs:
        if isinstance(tc_l, str):
            tc_l = [tc_l]

        for tc in tc_l:
            test_idx = len(test_cases) + 1

            use_timeout = command_timeout
            if first_command_timeout is not None and not commands:
                use_timeout = first_command_timeout
            commands.append(
                Command(
                    command=["python3", "main.py", str(test_idx)],
                    timeout=use_timeout,
                )
            )
            tc = APPS_IF_BRANCH.replace("%%%test_id%%%", str(test_idx)).replace(
                "%%%test_case%%%", tc
            )
            if test_idx != 1:
                tc = "el" + tc
            test_cases.append(tc)

        if max_commands and len(commands) >= max_commands:
            break

    test_cases = APPS_ASSERT_IF_TEMPLATE.replace(
        "%%%TEST_IFS%%%", "\n".join(test_cases)
    )
    files["main.py"] = files["main.py"] + "\n\n" + test_cases

    return Executable(
        files=files,
        commands=commands,
        tracked_files=[],
        ensure_all_run=False,
        should_early_stop=early_stop_fn,
    )


def postprocess_program_result(
    command_result: ExecutionResult,
    uses_stdin: bool,
    expected_out: List[str],
    expected_num_commands: int,
) -> Dict:
    """Processes the result of executing a program for APPs.

    Returns a dict with the following keys:
    - timed_out: Whether the program timed out.
    - had_error: Whether the program had an error.
    - passed: Whether the program passed.
    - return_code: The return code of the program.
    - stderr: The stderr of the program.
    - stdout: The stdout of the program.
    """
    if command_result.timed_out or not command_result.command_results:
        passed = False
    elif not command_result.all_had_return_code(0):
        passed = False
    elif len(command_result.command_results) < expected_num_commands:
        passed = False
    else:
        if uses_stdin:

            actual = [
                r.stdout.replace("\r", "")
                for r in command_result.command_results
            ]
            passed = actual == expected_out
        else:
            passed = True
    return {
        "timed_out": command_result.timed_out,
        "had_error": command_result.had_error,
        "passed": passed,
        "return_code": command_result.last_cmd.return_code,
        "stderr": [
            r.stderr.replace("\r", "") for r in command_result.command_results
        ],
        "stdout": [
            r.stdout.replace("\r", "") for r in command_result.command_results
        ],
    }


def preprocessor(
    problem,
    command_timeout: float = 2.0,
    first_command_timeout: Optional[float] = None,
    max_commands: int = None,
    early_stopping: bool = False,
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    max_memory: str = "4*1024*1024*1024",
) -> List[Executable]:
    out = []

    if "exec_mode" not in problem:
        problem = process_raw_example(problem)

    for sol in problem[solution_list_key]:
        out.append(
            make_executable(
                solution=sol,
                inputs=problem["inputs"],
                outputs=problem["outputs"],
                exec_mode=problem["exec_mode"],
                command_timeout=command_timeout,
                max_commands=max_commands,
                early_stopping=early_stopping,
                solution_str_key=solution_str_key,
                max_memory=max_memory,
                first_command_timeout=first_command_timeout,
            )
        )
    return out


def postprocessor(
    problem: Dict,
    result_list: List[ExecutionResult],
    max_commands: int = None,
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
) -> Dict:
    out = []
    if "exec_mode" not in problem:
        input_output = ujson.loads(problem["input_output"])
        uses_stdin = "fn_name" not in input_output
        outputs = [o for o in input_output["outputs"]]
        solutions = ujson.loads(problem[solution_list_key])
    else:
        uses_stdin = problem["exec_mode"] == "stdin"
        outputs = problem["outputs"]
        solutions = problem[solution_list_key]
    expected_out = None
    if uses_stdin:

        expected_out = [o for o in outputs]
        if max_commands:
            expected_out = expected_out[:max_commands]
    for res, pred in zip(result_list, solutions):
        proc_res = postprocess_program_result(
            command_result=res,
            uses_stdin=uses_stdin,
            expected_out=expected_out,
            expected_num_commands=res.expected_num_commands,
        )
        if isinstance(pred, str):
            pred = {solution_str_key: pred}
        out.append({**pred, **proc_res})

    return {
        **{k: problem[k] for k in problem if k != solution_list_key},
        "expected_output": expected_out,
        "predictions": out,
    }


def evaluate(
    predictions: List[Dict],
    num_workers: int,
    first_command_timeout: int = 10,
    command_timeout: float = 1.0,
    max_commands: int = None,
    early_stopping: bool = False,
    k_vals: List[int] = None,
    execution_kwargs: Dict = None,
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    max_memory: str = "4*1024*1024*1024",
) -> Tuple[Dict, List[Dict]]:

    results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(
            num_workers=num_workers, **(execution_kwargs or {})
        ),
        preprocessor=partial(
            preprocessor,
            first_command_timeout=first_command_timeout,
            command_timeout=command_timeout,
            max_commands=max_commands,
            early_stopping=early_stopping,
            solution_str_key=solution_str_key,
            solution_list_key=solution_list_key,
            max_memory=max_memory,
        ),
        postprocessor=partial(
            postprocessor,
            max_commands=max_commands,
            solution_list_key=solution_list_key,
            solution_str_key=solution_str_key,
        ),
        preproc_returns_list=True,
    )

    num_samples = 1
    pass_counts = []
    for r in results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {
        "percent_passed": sum(pc > 0 for pc in pass_counts) / len(pass_counts),
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
