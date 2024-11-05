""" Module for evaluating apps dataset. """

import json
import logging
from functools import partial
from typing import Dict, List, Tuple

from datasets import Dataset

from code_execution.data_structures import Command
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


def preprocessor(
    problem,
    timeout: int = 10,
    command_timeout: float = 2.0,
    max_commands: int = None,
) -> Executable:
    out = []

    input_output = json.loads(problem["input_output"])

    for sol in json.loads(problem["solutions"]):
        files = {"main.py": sol}
        if "fn_name" not in input_output:
            commands = []
            for i in input_output["inputs"]:
                if isinstance(i, list):
                    stdin = i
                else:
                    stdin = i.split("\n")
                commands.append(
                    Command(
                        command=["python", "main.py"],
                        timeout=command_timeout,
                        stdin=stdin,
                    )
                )
                if max_commands and len(commands) >= max_commands:
                    break
        else:
            test_code = []
            # Leetcode needs imports to be in the solution
            prog = (
                APPS_IMPORT_STR + "\n\n" + APPS_OUTPUT_CONVERTER + "\n\n" + sol
            )

            use_fn_name = input_output["fn_name"]
            if "class Solution" in sol:
                use_fn_name = "Solution()." + use_fn_name

            for i, o in zip(input_output["inputs"], input_output["outputs"]):

                test_code.append(make_test_case(use_fn_name, i, o))

            files["main.py"] = prog + "\n\n" + "\n".join(test_code)
            commands = [
                Command(
                    command=["python", "main.py"],
                    timeout=timeout,
                )
            ]
        out.append(
            Executable(
                files=files,
                commands=commands,
                tracked_files=[],
                ensure_all_run=False,
            )
        )
    return out


def postprocessor(
    problem: Dict, result_list: List[ExecutionResult], max_commands: int = None
) -> Dict:
    out = []
    uses_stdin = "fn_name" not in problem["input_output"]
    expected_out = None
    if uses_stdin:
        expected_out = [o for o in problem["input_output"]["outputs"]]
        if max_commands:
            expected_out = expected_out[:max_commands]

    for res, pred in zip(result_list, problem["solutions"]):
        if res.had_error or res.timed_out:
            passed = False
        elif uses_stdin:
            if len(res.command_results) < len(expected_out):
                passed = False
            else:
                actual = [
                    r.stdout.replace("\r", "") for r in res.command_results
                ]
                passed = actual == expected_out

        else:
            passed = True

        out.append(
            {
                "solution": pred,
                "passed": passed,
                "return_code": res.last_cmd.return_code,
                "stderr": [
                    r.stderr.replace("\r", "") for r in res.command_results
                ],
                "stdout": [
                    r.stdout.replace("\r", "") for r in res.command_results
                ],
            }
        )

    return {
        **{k: problem[k] for k in problem if k != "solutions"},
        "predictions": out,
    }


def evaluate_apps(
    dataset: Dataset,
    num_workers: int,
    timeout: int = 10,
    command_timeout: float = 2.0,
    max_commands: int = None,
    k_vals: List[int] = None,
    execution_kwargs: Dict = None,
) -> Tuple[Dict, List[Dict]]:

    results = execute_predictions(
        pred_list=dataset,
        config=ExecutionConfig(
            num_workers=num_workers, **(execution_kwargs or {})
        ),
        preprocessor=partial(
            preprocessor,
            timeout=timeout,
            command_timeout=command_timeout,
            max_commands=max_commands,
        ),
        postprocessor=partial(postprocessor, max_commands=max_commands),
        preproc_returns_list=True,
    )

    num_samples = 1
    pass_counts = []
    for r in results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {}
    for k in k_vals or [1, 5, 10]:
        if k > num_samples:
            break
        metrics[f"pass@{k}"] = float(
            estimate_pass_at_k(
                [num_samples] * len(pass_counts), pass_counts, k
            ).mean()
        )

    return metrics, results
