""" Module for evaluating apps dataset. """

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import ujson

from code_execution.data_structures import Command
from code_execution.data_structures import CommandResult
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig

logger = logging.getLogger(__name__)

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


def make_test_case(fn_name, inputs, outputs):
    if isinstance(outputs, list) and len(outputs) == 1:
        expected_out = repr(outputs[0])
    else:
        expected_out = repr(outputs)
    return f"assert convert_output({fn_name}({','.join(map(repr,inputs))})) == {expected_out}"


def process_raw_example(example):
    try:
        input_output = ujson.loads(example["input_output"])
    except (ujson.JSONDecodeError, ValueError):
        input_output = {"inputs": [], "outputs": []}

    try:
        solutions = ujson.loads(example["solutions"])
    except ujson.JSONDecodeError:
        solutions = []

    # check if the solution is a class, so we can call the function Solution.fn_name
    is_cls = any("class Solution" in s for s in solutions)

    # convert these to strings for pyarrow
    new_inputs = []
    new_outputs = []
    for inp, out in zip(input_output["inputs"], input_output["outputs"]):

        if "fn_name" in input_output:
            if is_cls:
                use_fn_name = "Solution()." + input_output["fn_name"]
            else:
                use_fn_name = input_output["fn_name"]
            new_inputs.append([make_test_case(use_fn_name, inp, out)])
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
        "exec_mode": "asserts" if "fn_name" in input_output else "stdin",
    }


def should_stop_early(
    cmd_idx: int, res: CommandResult, is_stdin: bool, expected_out: List[str]
) -> bool:
    if res.had_error or res.timed_out:
        return True

    if not is_stdin:
        return False
    if len(expected_out) <= cmd_idx:
        return True

    if res.stdout.replace("\r", "") != expected_out[cmd_idx]:
        return True

    return False


def make_executable(
    solution: str,
    inputs: List[List[str]],
    outputs: List[str],
    exec_mode: str,
    timeout: int = 10,
    command_timeout: float = 2.0,
    max_commands: int = None,
    early_stopping: bool = False,
) -> List[Executable]:
    files = {"main.py": solution}
    if exec_mode == "stdin":
        commands = []
        for i in inputs:
            commands.append(
                Command(
                    command=["python", "main.py"],
                    timeout=command_timeout,
                    stdin=i,
                )
            )
            if max_commands and len(commands) >= max_commands:
                break
    elif exec_mode == "asserts":
        # Leetcode needs imports to be in the solution
        prog = (
            APPS_IMPORT_STR + "\n\n" + APPS_OUTPUT_CONVERTER + "\n\n" + solution
        )
        files["main.py"] = prog + "\n\n" + "\n".join(inputs[0])
        commands = [
            Command(
                command=["python", "main.py"],
                timeout=timeout,
            )
        ]
    else:
        raise ValueError(f"Unknown exec_mode: {exec_mode}")
    early_stop_fn = None
    if early_stopping:
        early_stop_fn = partial(
            should_stop_early,
            expected_out=outputs,
            is_stdin=exec_mode == "stdin",
        )
    return Executable(
        files=files,
        commands=commands,
        tracked_files=[],
        ensure_all_run=False,
        should_early_stop=early_stop_fn,
    )


def postprocess_program_result(
    pred: Dict,
    command_result: ExecutionResult,
    uses_stdin: bool,
    expected_out: List[str],
) -> bool:
    if command_result.had_error or command_result.timed_out:
        passed = False
    elif uses_stdin:
        if len(command_result.command_results) < len(expected_out):
            passed = False
        else:
            actual = [
                r.stdout.replace("\r", "")
                for r in command_result.command_results
            ]
            passed = actual == expected_out
    else:
        passed = True

    return {
        **pred,
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
    timeout: int = 10,
    command_timeout: float = 2.0,
    max_commands: int = None,
    early_stopping: bool = False,
) -> Executable:
    out = []

    if "exec_mode" not in problem:
        problem = process_raw_example(problem)

    for sol in problem["solutions"]:
        out.append(
            make_executable(
                solution=sol,
                inputs=problem["inputs"],
                outputs=problem["outputs"],
                exec_mode=problem["exec_mode"],
                timeout=timeout,
                command_timeout=command_timeout,
                max_commands=max_commands,
                early_stopping=early_stopping,
            )
        )
    return out


def postprocessor(
    problem: Dict, result_list: List[ExecutionResult], max_commands: int = None
) -> Dict:
    out = []
    if "exec_mode" not in problem:
        input_output = ujson.loads(problem["input_output"])
        uses_stdin = "fn_name" not in input_output
        outputs = [o for o in input_output["outputs"]]
        solutions = ujson.loads(problem["solutions"])
    else:
        uses_stdin = problem["exec_mode"] == "stdin"
        outputs = problem["outputs"]
        solutions = problem["solutions"]
    expected_out = None
    if uses_stdin:

        expected_out = [o for o in outputs]
        if max_commands:
            expected_out = expected_out[:max_commands]

    for res, pred in zip(result_list, solutions):
        out.append(
            postprocess_program_result(
                command_result=res,
                pred=pred,
                uses_stdin=uses_stdin,
                expected_out=expected_out,
            )
        )

    return {
        **{k: problem[k] for k in problem if k != "solutions"},
        "expected_output": expected_out,
        "predictions": out,
    }


def evaluate(
    predictions: List[Dict],
    num_workers: int,
    timeout: int = 10,
    command_timeout: float = 2.0,
    max_commands: int = None,
    early_stopping: bool = False,
    k_vals: List[int] = None,
    execution_kwargs: Dict = None,
) -> Tuple[Dict, List[Dict]]:

    results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(
            num_workers=num_workers, **(execution_kwargs or {})
        ),
        preprocessor=partial(
            preprocessor,
            timeout=timeout,
            command_timeout=command_timeout,
            max_commands=max_commands,
            early_stopping=early_stopping,
        ),
        postprocessor=partial(
            postprocessor,
            max_commands=max_commands,
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
