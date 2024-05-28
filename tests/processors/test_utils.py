import math
from collections import Counter
from dataclasses import asdict

import pytest

from code_execution.processors import utils as processor_utils


@pytest.mark.parametrize("test_1_result", ["PASSED", "FAILED", "A", "MISSING"])
@pytest.mark.parametrize("test_2_result", ["PASSED", "FAILED", "B", "MISSING"])
def test_parse_tc_stdout(test_1_result, test_2_result):
    stdout = ""
    if test_1_result != "MISSING":
        stdout += f"TEST_1___{test_1_result}\n"
    if test_2_result != "MISSING":
        stdout += f"TEST_2___{test_2_result}\n"

    actual = processor_utils.parse_tc_stdout(
        stdout, expected_test_cases={"1", "2"}
    )
    assert actual.test_results == {"1": test_1_result, "2": test_2_result}
    assert not actual.unknown_tests
    assert not actual.duplicated_tests
    expected_pct_pass = 0
    expected_num_passed = 0
    if test_1_result == "PASSED":
        expected_pct_pass += 0.5
        expected_num_passed += 1
    if test_2_result == "PASSED":
        expected_pct_pass += 0.5
        expected_num_passed += 1
    assert math.isclose(actual.percent_passed, expected_pct_pass, rel_tol=1e-9)
    assert actual.num_passed == expected_num_passed

    error_counts = {}
    should_be_missing = False
    if test_1_result not in ["PASSED", "FAILED"]:
        if test_1_result == "MISSING":
            should_be_missing = True
        else:
            error_counts[test_1_result] = 1
    if test_2_result not in ["PASSED", "FAILED"]:
        if test_2_result == "MISSING":
            should_be_missing = True
        elif test_2_result not in error_counts:
            error_counts[test_2_result] = 1
        else:
            error_counts[test_2_result] += 1
    assert actual.error_counts == error_counts
    assert actual.is_missing_tests == should_be_missing


def test_parse_tc_stdout_skip_lines():
    stdout = "TEST_1___PASSED\nTEST_0.PASSED\nTEST_2___FAILED\n"
    actual = processor_utils.parse_tc_stdout(
        stdout, expected_test_cases={"1", "2"}
    )
    assert actual.test_results == {"1": "PASSED", "2": "FAILED"}
    assert not actual.unknown_tests
    assert not actual.duplicated_tests
    assert actual.num_passed == 1
    assert math.isclose(actual.percent_passed, 0.5, rel_tol=1e-9)
    assert not actual.error_counts
    assert not actual.is_missing_tests


def test_parse_tc_stdout_duplicates():
    stdout = "TEST_1___PASSED\nTEST_1___FAILED\nTEST_2___PASSED\n"
    actual = processor_utils.parse_tc_stdout(
        stdout,
        expected_test_cases={"1", "2"},
        did_test_pass=lambda x, y: x == "PASSED"
        and all(z == "PASSED" for z in y),
    )
    assert actual.test_results == {"1": "PASSED", "2": "PASSED"}
    assert not actual.unknown_tests
    assert actual.duplicated_tests == {"1": ["FAILED"]}
    assert actual.num_passed == 1
    assert math.isclose(actual.percent_passed, 0.5, rel_tol=1e-9)


def test_parse_tc_stdout_unkown():
    stdout = "TEST_0___PASSED\nTEST_1___FAILED\nTEST_2___PASSED\n"
    actual = processor_utils.parse_tc_stdout(
        stdout,
        expected_test_cases={"1", "2"},
    )
    assert actual.test_results == {"1": "FAILED", "2": "PASSED"}
    assert actual.unknown_tests == {"0": "PASSED"}
    assert not actual.duplicated_tests
    assert actual.num_passed == 1
    assert math.isclose(actual.percent_passed, 0.5, rel_tol=1e-9)


@pytest.mark.parametrize(
    "test_results,expected_outcome",
    [
        ({"1": "PASSED", "2": "PASSED"}, "PASSED"),
        ({"1": "PASSED", "2": "FAILED"}, "FAILED"),
        ({}, "COMPILE_ERROR"),
        ({"1": "PASSED"}, "FAILED"),
        ({"1": "AssertionError", "2": "TypeError"}, "FAILED"),
        ({"1": "MISSING", "2": "MISSING"}, "TIMED_OUT"),
    ],
    ids=[
        "PASSED",
        "FAILED",
        "COMPILE_ERROR",
        "FAILED_W_MISSING",
        "ALL_ERROR",
        "TIMED_OUT",
    ],
)
def test_get_prediction_outcome(test_results, expected_outcome):
    num_passed = sum(1 for v in test_results.values() if v == "PASSED")

    parsed_result = processor_utils.ParsedTestResults(
        test_results=test_results,
        duplicated_tests={},
        unknown_tests={},
        error_counts={},
        is_missing_tests=len(test_results) < 2,
        num_expected_tests=2,
        num_passed=num_passed,
    )

    outcome = processor_utils.get_prediction_outcome(
        timed_out=expected_outcome == "TIMED_OUT",
        parsed_result=parsed_result,
        return_code=expected_outcome == "COMPILE_ERROR",
    )
    assert outcome == processor_utils.PredictionOutcome[expected_outcome]


@pytest.mark.parametrize(
    "allow_duplicates", [True, False], ids=["allow_duplicates", "no_duplicates"]
)
@pytest.mark.parametrize(
    "allow_unknown_tests",
    [True, False],
    ids=["allow_unknown_tests", "no_unknown_tests"],
)
def test_get_prediction_outcome_flags(allow_duplicates, allow_unknown_tests):
    parsed_result = processor_utils.ParsedTestResults(
        test_results={"1": "PASSED", "2": "PASSED"},
        duplicated_tests={"1": ["FAILED"]},
        unknown_tests={"3": "PASSED"},
        error_counts={},
        is_missing_tests=False,
        num_expected_tests=2,
        num_passed=2,
    )

    outcome = processor_utils.get_prediction_outcome(
        parsed_result=parsed_result,
        timed_out=False,
        return_code=0,
        allow_duplicates=allow_duplicates,
        allow_unknown_tests=allow_unknown_tests,
    )
    if allow_duplicates and allow_unknown_tests:
        expected_out = processor_utils.PredictionOutcome.PASSED
    elif not allow_duplicates:
        expected_out = processor_utils.PredictionOutcome.DUPLICATE_TESTS
    elif not allow_unknown_tests:
        expected_out = processor_utils.PredictionOutcome.UNKNOWN_TESTS
    assert outcome == expected_out


@pytest.mark.parametrize(
    "timed_out", [True, False], ids=["timed_out", "not_timed_out"]
)
def test_get_prediction_outcome_missing_all(timed_out):
    parsed_result = processor_utils.ParsedTestResults(
        test_results={},
        duplicated_tests={},
        unknown_tests={},
        error_counts={"MISSING": 2},
        is_missing_tests=False,
        num_passed=2,
        num_expected_tests=2,
        all_error=False,
    )
    outcome = processor_utils.get_prediction_outcome(
        parsed_result, 0, timed_out=timed_out
    )
    expected = (
        processor_utils.PredictionOutcome.TIMED_OUT
        if timed_out
        else processor_utils.PredictionOutcome.COMPILE_ERROR
    )
    assert outcome == expected


def test_get_prediction_outcome_with_missing():
    parsed_result = processor_utils.ParsedTestResults(
        test_results={"0": "PASSED"},
        duplicated_tests={},
        unknown_tests={},
        error_counts={},
        is_missing_tests=True,
        num_passed=2,
        num_expected_tests=2,
        all_error=False,
    )

    outcome = processor_utils.get_prediction_outcome(
        parsed_result, 0, timed_out=False
    )
    expected = processor_utils.PredictionOutcome.FAILED
    assert outcome == expected
