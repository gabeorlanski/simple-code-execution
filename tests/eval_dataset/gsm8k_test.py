import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from code_execution.data_structures import CommandResult
from code_execution.data_structures import ExecutionResult
from code_execution.eval_dataset import gsm8k
from code_execution.eval_dataset.metrics import estimate_pass_at_k


# Test case fixtures to improve readability
@pytest.fixture
def successful_result():
    return ExecutionResult(
        key="test1",
        command_results=[
            CommandResult(
                return_code=0,
                runtime=0.1,
                stdout="Hello",
                stderr="",
                timed_out=False,
            )
        ],
        elapsed=0.2,
        cwd="/tmp",
        tracked_files={},
        expected_num_commands=1,
        writing_time=0.05,
        cleanup_time=0.03,
        preprocess_time=0.02,
    )


def test_postprocess_result(successful_result):
    actual = gsm8k.postprocess_program_result(
        {"solution": "test"},
        successful_result,
    )

    assert actual["solution"] == "test"
    assert actual["passed"]

    assert len(actual["timing"]["cmd_eval"]) == 1
    assert actual["timing"]["cmd_exec"] == [0.1]
    assert actual["stdout"] == "Hello"

    assert actual["stderr"] == ""
    assert not actual["timeout"]
    assert not actual["had_error"]
    assert actual["return_code"] == 0
    assert actual["timing"]["writing"] == successful_result.writing_time
    assert actual["timing"]["cleanup"] == successful_result.cleanup_time
    assert actual["timing"]["preprocess"] == successful_result.preprocess_time
    assert actual["timing"]["execution"] == successful_result.elapsed


def test_evaluate():
    problems = [
        {
            "solutions": [
                {
                    "cid": 0,
                    "solution": "def solution():\n    return 1",
                    "should_pass": True,
                },
                {
                    "cid": 1,
                    "solution": "def solution():\n    return 1",
                    "should_pass": True,
                },
                {
                    "cid": 2,
                    "solution": "def solution():\n    return 1",
                    "should_pass": True,
                },
                {
                    "cid": 3,
                    "solution": "def solution():\n    return 2",
                    "should_pass": False,
                },
            ],
            "answer": "$1",
        },
        {
            "solutions": [
                {
                    "should_pass": True,
                    "cid": 0,
                    "solution": "def solution():\n    return 5000",
                },
                {
                    "should_pass": False,
                    "cid": 1,
                    "solution": "def solution():\n    return 6000",
                },
                {
                    "should_pass": False,
                    "cid": 2,
                    "solution": "def solution():\n    return 7000",
                },
                {
                    "should_pass": False,
                    "cid": 3,
                    "solution": "def solution():\n    return 7000",
                },
            ],
            "answer": "5,0 00",
        },
        {
            "solutions": [
                {
                    "should_pass": True,
                    "cid": 0,
                    "solution": "def solution():\n    return 1",
                },
                {
                    "should_pass": True,
                    "cid": 1,
                    "solution": "def solution():\n    return 1",
                },
                {
                    "should_pass": True,
                    "cid": 2,
                    "solution": "def solution():\n    return 1",
                },
                {
                    "should_pass": True,
                    "cid": 3,
                    "solution": "def solution():\n    return 1",
                },
            ],
            "answer": "1",
        },
        {
            "solutions": [
                {
                    "should_pass": False,
                    "cid": 0,
                    "solution": "def solution():\n    return 0",
                },
                {
                    "should_pass": False,
                    "cid": 1,
                    "solution": "def solution():\n    return Nashj",
                },
                {
                    "should_pass": False,
                    "cid": 2,
                    "solution": "def solution():\n    return 1-None",
                },
                {
                    "should_pass": False,
                    "cid": 3,
                    "solution": "def solution():\n    return 10",
                },
            ],
            "answer": "5,0 00",
        },
    ]

    metrics, preds = gsm8k.evaluate(
        problems,
        1,
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

    expected = 0.75

    expected_pass_at_k = estimate_pass_at_k(
        [4, 4, 4, 4], [3, 1, 4, 0], 1
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
            assert actual_sol["passed"] == expected_sol["should_pass"]
            assert "timing" in actual_sol
            assert set(actual_sol["timing"].keys()) == {
                "writing",
                "cleanup",
                "cmd_exec",
                "cmd_eval",
                "preprocess",
                "execution",
            }
