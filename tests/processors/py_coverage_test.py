import json
import math
from pathlib import Path
from unittest import mock

import pytest

from code_execution.execution import serial_execute_code
from code_execution.processors.python import py_coverage
from code_execution.processors.utils import get_processor


@pytest.fixture()
def preprocessor():
    return get_processor("py_coverage")[0]


@pytest.fixture()
def postprocessor():
    return get_processor("py_coverage")[1]


def test_preprocess(preprocessor):
    imports, context, tracked_files = preprocessor("TEST_RUNNER")

    assert imports == py_coverage.IMPORTS
    assert context == py_coverage.RUNNER_TEMPLATE.format(
        runner_call="TEST_RUNNER"
    )
    assert tracked_files == ["coverage.json"]


@pytest.mark.parametrize("file_name", ["solution.py", "test.py"])
def test_postprocess_(file_name, postprocessor):
    coverage = {
        "solution.py": {
            "summary": {"covered_lines": 10, "percent_covered": 100},
            "missing_lines": [1, 2, 3],
        },
        "test.py": {
            "summary": {"covered_lines": 5, "percent_covered": 2},
            "missing_lines": [4, 5, 6],
        },
    }

    result = postprocessor(
        {
            "coverage.json": json.dumps(
                {"files": coverage, "not important": {"main.py": {}}}
            )
        },
        solution_file_name=file_name,
    )

    assert (
        result["percent_covered"]
        == coverage[file_name]["summary"]["percent_covered"]
    )
    assert result["missing_lines"] == len(coverage[file_name]["missing_lines"])
