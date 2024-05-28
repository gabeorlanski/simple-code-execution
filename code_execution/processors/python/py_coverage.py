""" Module for processing tests and solutions for running with coverage. """

import json
from typing import Dict

from ..utils import register_postprocessor
from ..utils import register_preprocessor

RUNNER_TEMPLATE = """cov = coverage.Coverage()
cov.start()
{runner_call}
cov.stop()
cov.save()
cov.json_report(show_contexts=True)"""

IMPORTS = "import coverage"


@register_preprocessor("py_coverage")
def preprocess(
    runner_call: str,
):
    return (
        IMPORTS,
        RUNNER_TEMPLATE.format(runner_call=runner_call),
        ["coverage.json"],
    )


EMPTY_RESULT = {
    "percent_covered": 0,
    "missing_lines": -1,
}


@register_postprocessor("py_coverage")
def postprocess(
    tracked_files: Dict[str, str],
    solution_file_name: str,
) -> Dict:
    coverage_result = tracked_files.get("coverage.json")

    if coverage_result is None:
        return EMPTY_RESULT
    try:
        coverage = json.loads(tracked_files["coverage.json"])
    except json.JSONDecodeError:
        return EMPTY_RESULT
    if "files" not in coverage:
        return EMPTY_RESULT

    if solution_file_name not in coverage["files"]:
        return EMPTY_RESULT
    solution_coverage = coverage["files"][solution_file_name]
    return {
        "percent_covered": solution_coverage["summary"]["percent_covered"],
        "missing_lines": len(solution_coverage["missing_lines"]),
    }
