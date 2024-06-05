import enum
import math
import re
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, List, Set, Tuple, Union


class PredictionOutcome(enum.Enum):
    PASSED = enum.auto()
    FAILED = enum.auto()
    COMPILE_ERROR = enum.auto()
    DUPLICATE_TESTS = enum.auto()
    UNKNOWN_TESTS = enum.auto()
    TIMED_OUT = enum.auto()
    MISSING_INFO = enum.auto()


PASS_STR = PredictionOutcome.PASSED.name
FAIL_STR = PredictionOutcome.FAILED.name
EXCEPTION_STR = "type(e).__name__"
DEFAULT_TEST_DELIMITER = "___"
TEST_PRINT_STR = ""
FORMAT_ERROR_MSG = "TEST_FORMAT_ERROR"
TEST_DELIM_REGEX = re.compile(
    rf"^TEST_(\d+){re.escape(DEFAULT_TEST_DELIMITER)}(.*)"
)


PREPROCESSORS = {}
POSTPROCESSORS = {}


def register_preprocessor(name):
    """Registers a processor."""

    def decorator(fn):
        PREPROCESSORS[name] = fn
        return fn

    return decorator


def register_postprocessor(name):
    """Registers a processor."""

    def decorator(fn):
        POSTPROCESSORS[name] = fn
        return fn

    return decorator


def get_preprocessor(name):
    """Gets a processor by name."""
    return PREPROCESSORS[name]


def get_postprocessor(name):
    """Gets a processor by name."""
    return POSTPROCESSORS[name]


def get_processor(name):
    """Gets a processor by name."""
    return PREPROCESSORS[name], POSTPROCESSORS[name]


@dataclass
class ParsedTestResults:
    test_results: Dict[str, Union[str, Tuple, int]]
    error_counts: Dict
    is_missing_tests: bool
    num_passed: int
    num_expected_tests: int
    all_error: bool = False
    unknown_tests: Dict[str, Union[str, Tuple, int]] = field(
        default_factory=dict
    )
    duplicated_tests: Dict[str, List[Union[str, Tuple, int]]] = field(
        default_factory=dict
    )

    def to_dict(self):
        out = asdict(self)
        out["error_counts"] = dict(self.error_counts)
        return out

    @property
    def percent_passed(self):
        """Calculates the percent of tests passed."""
        return self.num_passed / self.num_expected_tests

    @property
    def percent_error(self):
        """Calculates the percent of tests with errors.

        An error is a test that did not pass or have an assertion error."""
        num_error = sum(
            map(
                lambda x: x not in [PASS_STR, FAIL_STR],
                self.test_results.values(),
            )
        )
        return num_error / self.num_expected_tests

    @property
    def errored_tests(self) -> Dict[str, str]:
        """Gets the tests that errored."""
        return {
            k: v
            for k, v in self.test_results.items()
            if v not in {PASS_STR, FAIL_STR}
        }


def _default_did_test_pass(result: str, *_) -> bool:
    return result == PASS_STR


def parse_tc_stdout(
    stdout: str,
    expected_test_cases: Set[str],
    parsing_regex: re.Pattern = TEST_DELIM_REGEX,
    idx_processor: Callable[[str], str] = lambda x: x,
    result_processor: Callable[[str], Union[str, Tuple, int]] = lambda x: x,
    did_test_pass: Callable[[Any, List[Any]], bool] = _default_did_test_pass,
) -> Tuple[float, ParsedTestResults]:
    """Parses the stdout from execution and returns the mapping of idx->[result].

    It splits `stdout` into lines and then uses `parsing_regex` to parse each
    line. If no match is found, the line is ignored. If a match is found, the
    first capturing group is used as the test idx and the second as the result.

    Args:
        stdout: The stdout from the execution.
        parsing_regex: The regex to use to parse the stdout. Must have one capturing group for the test idx and one for the result.
        idx_processor: A function to process the idx from the regex. Defaults to `lambda x: x`.
        result_processor: A function to process the result from the regex. Defaults to `lambda x: x`.
        did_test_pass: A function to determine if a test passed. Defaults to `result == PASSED`.

    Returns:
        The percent of tests passed and the parsed test results.

    """
    test_results = {}
    unknown_tests = {}
    duplicated_tests = defaultdict(list)
    error_counts = Counter()
    num_errors = 0
    had_non_error = False
    for line in stdout.split("\n"):
        match = parsing_regex.match(line)
        if match is None:
            continue

        idx = idx_processor(match.group(1))
        result = result_processor(match.group(2))
        is_unknown = idx not in expected_test_cases
        if idx in test_results or idx in unknown_tests:
            duplicated_tests[idx].append(result)
            continue
        if is_unknown:
            unknown_tests[idx] = result
            continue
        test_results[idx] = result
        if result not in [PASS_STR, FAIL_STR]:
            error_counts[result] += 1
            num_errors += 1
        else:
            had_non_error = True
    is_missing_tests = False

    num_passed = 0
    for k in expected_test_cases:
        if k not in test_results and k not in unknown_tests:
            test_results[k] = "MISSING"
            is_missing_tests = True
        else:
            num_passed += did_test_pass(
                test_results[k], duplicated_tests.get(k, [])
            )
    parsed_tests = ParsedTestResults(
        test_results=test_results,
        unknown_tests=unknown_tests,
        duplicated_tests=dict(duplicated_tests),
        error_counts=error_counts,
        is_missing_tests=is_missing_tests,
        num_passed=num_passed,
        num_expected_tests=len(expected_test_cases),
        all_error=not had_non_error,
    )
    return parsed_tests


def get_prediction_outcome(
    parsed_result: ParsedTestResults,
    return_code: int,
    timed_out: bool,
    allow_duplicates: bool = False,
    allow_unknown_tests: bool = False,
) -> PredictionOutcome:
    """Gets the percent passed and the overall outcome for a prediction.

    Args:
        parsed_results: The parsed test results.
        timed_out: Whether the prediction timed out.
        allow_duplicates: Whether to allow duplicated tests.
        allow_unknown_tests: Whether to allow unknown tests.

    Returns:
        The outcome for the prediction.
    """

    if return_code != 0:
        return PredictionOutcome.COMPILE_ERROR
    if timed_out:
        return PredictionOutcome.TIMED_OUT

    if parsed_result.duplicated_tests and not allow_duplicates:
        return PredictionOutcome.DUPLICATE_TESTS
    if parsed_result.unknown_tests and not allow_unknown_tests:
        return PredictionOutcome.UNKNOWN_TESTS
    if set(parsed_result.test_results.values()) == {"MISSING"}:
        return PredictionOutcome.COMPILE_ERROR
    if not parsed_result.test_results:
        return PredictionOutcome.COMPILE_ERROR
    if (
        math.isclose(parsed_result.percent_passed, 1, rel_tol=1e-9)
        and not parsed_result.is_missing_tests
    ):
        return PredictionOutcome.PASSED

    return PredictionOutcome.FAILED
