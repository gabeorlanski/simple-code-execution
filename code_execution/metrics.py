""" Metrics for evaluating the performance of the code execution. """

import ast
import logging
from collections import Counter

import numpy as np

from code_execution.code_trees import safe_ast_parse

logger = logging.getLogger(__name__)


def naive_process_result(result):
    """The most naive way to process the result."""
    if result["failed"] or result["timed_out"]:
        return False
    return True


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calc_question_metric(question_predictions):
    """Calculate question metrics from the predictions."""
    # The naming may not be optimal, but the difference is that one is for the
    # question as a whole (i.e. was the question solved). The other dict is for
    # metrics that describe the individual predictions for the question (i.e.
    # the number of correct samples).
    question_stats = {
        "num_passed": 0,
        "num_syntax_errors": 0,
        "result_count": Counter(),
    }
    pred_stats = {
        "num_lines": [],
    }

    # Used for sanity check
    num_test_cases = set()

    was_question_solved = False
    for pred in question_predictions:
        if pred["passed"]:
            was_question_solved = True
        question_stats["num_passed"] += pred["passed"]
        code = pred["solution"].strip()
        tree = safe_ast_parse(code)
        num_test_cases.add(len(pred["test_cases"]))
        if tree is None:
            question_stats["num_syntax_errors"] += 1
        else:
            code = ast.unparse(tree).strip()

        question_stats["result_count"][pred["outcome"]] += 1
        pred_stats["num_lines"].append(len(code.splitlines()))

    if len(num_test_cases) != 1:
        logger.error(question_predictions)
        raise ValueError("Different number of tests.")
    pred_stats["num_lines"] = np.mean(pred_stats["num_lines"])
    return was_question_solved, question_stats, pred_stats
