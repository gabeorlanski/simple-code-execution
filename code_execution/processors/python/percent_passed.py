""" Module for the percent passed processor. """

import ast
import re
from typing import List, Optional, Set, Tuple, Union

from ...code_trees import convert_call_to_assert
from ...execution import CommandResult
from ..utils import FAIL_STR
from ..utils import PASS_STR
from ..utils import ParsedTestResults
from ..utils import PredictionOutcome
from ..utils import get_prediction_outcome
from ..utils import parse_tc_stdout
from ..utils import register_postprocessor
from ..utils import register_preprocessor

MEMORY_REGEX = re.compile(r"__MEMORY_USED__=(\d+)")
RUNTIME_REGEX = re.compile(r"__RUNTIME__=(\d*\.?\d+)")

DEFAULT_IMPORTS = """import signal
import tracemalloc
from typing import Optional
import time
import inspect
"""

CALL_CODE = """tracemalloc.start() 
t0 = time.time()
run_tests(test_timeout={test_timeout})
t1 = time.time()
print("__MEMORY_USED__="+str(tracemalloc.get_traced_memory()[1]))
tracemalloc.stop()
print("__RUNTIME__="+str(t1-t0))
"""

TEST_FUNC_TEMPLATE = "@test_wrapper\ndef test_{idx}({entry_point}):\n\tpass"
RUNNER_TEMPLATE = """def run_tests(test_timeout:Optional[int]=None):
    from {solution_module} import {entry_point}"""


WRAPPER_CONTEXT = f"""class PredTimeoutError(Exception):
    \"\"\"Timeout error for running commands.\"\"\"
    
def signal_handler(signum, frame):
    raise PredTimeoutError("Timed out!")

def test_wrapper(func):
    test_name = func.__name__.upper()
    def inner(*args,test_timeout:Optional[int]=None,**kwargs):
        try:
            if test_timeout is not None:
                signal.setitimer(signal.ITIMER_REAL, test_timeout)
                signal.signal(signal.SIGALRM, signal_handler)
            func(*args,**kwargs)
            result = "{PASS_STR}"
        except AssertionError:
            result = "{FAIL_STR}"
        except Exception as e:
            mod = inspect.getmodule(e)
            result = type(e).__name__
            if mod is not None and mod != '__main__':
                result = mod.__name__ + "." + result
        finally:
            if test_timeout is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
        print(test_name+"___"+result)
    return inner"""


def _make_test_function(
    idx: int,
    entry_point: str,
    test_case: Union[Tuple[str, str, bool], str, ast.AST],
) -> str:
    tree = ast.parse(
        TEST_FUNC_TEMPLATE.format(idx=idx, entry_point=entry_point)
    ).body[0]
    if isinstance(test_case, (tuple, list)):
        call, expected, requires_float = test_case

        test_tree = convert_call_to_assert(
            call=call,
            expected_output=expected,
            requires_float=requires_float,
        )
    elif isinstance(test_case, ast.AST):
        test_tree = test_case
    else:
        test_tree = ast.parse(test_case)

    test_tree = ast.copy_location(test_tree, tree.body[0])
    tree.body = test_tree.body
    return ast.unparse(tree)


def _make_test_runner(
    num_test_cases: int,
    module_name: str,
    entry_point: str,
):
    tree = ast.parse(
        RUNNER_TEMPLATE.format(
            solution_module=module_name, entry_point=entry_point
        )
    )

    test_calls = [
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id=f"test_{idx}", ctx=ast.Load()),
                args=[ast.Name(id=entry_point, ctx=ast.Load())],
                keywords=[
                    ast.keyword(
                        arg="test_timeout",
                        value=ast.Name(id="test_timeout", ctx=ast.Load()),
                    ),
                ],
            )
        )
        for idx in range(num_test_cases)
    ]

    tree.body[0].body += test_calls
    tree.body = list(map(ast.fix_missing_locations, tree.body))
    return ast.unparse(tree)


@register_preprocessor("py_percent_passed")
def preprocess(
    entry_point: str,
    test_cases: Union[Tuple[str, str, bool], str, List[ast.AST]],
    tc_imports: Optional[str] = None,
    tc_timeout: Optional[int] = None,
    entry_file: str = "solution",
) -> Tuple[str, str, str]:
    """Preprocess the prediction for a percent passed execution.

    Args:
        entry_point (str): The entry point of the function to use.
        test_cases (Union[Tuple[str, str, bool], str, List[ast.AST]]): The test cases to use.
        tc_imports (Optional[str], optional): The imports for the test cases. Defaults to None.
        tc_timeout (Optional[int], optional): The timeout for each individual test case. Defaults to None.
        entry_file (str, optional): The name of the entry file. Defaults to "solution".

    Returns:
        Tuple[str, str, str]: Imports, context, and the call code.
    """
    imports = DEFAULT_IMPORTS
    if tc_imports is not None:
        imports += "\n"
        if isinstance(tc_imports, list):
            tc_imports = "\n".join(tc_imports)
        imports += tc_imports

    tests = []
    for idx, test_case in enumerate(test_cases):
        tests.append(_make_test_function(idx, entry_point, test_case))
    if not tests:
        return "", "", ""

    test_runner = _make_test_runner(
        len(test_cases),
        entry_file,
        entry_point,
    )

    context = WRAPPER_CONTEXT
    context += "\n\n" + "\n\n".join(tests) + "\n\n" + test_runner

    call_code = CALL_CODE.format(test_timeout=tc_timeout)

    return imports, context, call_code


@register_postprocessor("py_percent_passed")
def postprocess(
    result: CommandResult,
    expected_test_cases: Set[str],
    allow_duplicates: bool = False,
    allow_unknown_tests: bool = False,
    expecting_memory: bool = True,
) -> Tuple[PredictionOutcome, ParsedTestResults, int, float]:
    """Postprocesses the result of a percent passed execution.

    Args:
        result (CommandResult): The execution result for the percent passed.
        expected_test_cases (Set[str]): The set of indices of expected test cases.
        allow_duplicates (bool, optional): Allow execution to return multiple results for single test cases. Defaults to False.
        allow_unknown_tests (bool, optional): Allow results to have tests not in expected. Defaults to False.
        expecting_memory (bool, optional): Should the postprocessor expect to find memory usage information. Defaults to True.

    Returns:
        Tuple[PredictionOutcome, ParsedTestResults, int, float]:
            The outcome of the prediction, the parsed test results, the memory used, and the runtime.
    """
    parsed_test_cases = parse_tc_stdout(
        result.stdout, expected_test_cases=expected_test_cases
    )

    outcome = get_prediction_outcome(
        parsed_test_cases,
        return_code=result.return_code,
        timed_out=result.timed_out,
        allow_duplicates=allow_duplicates,
        allow_unknown_tests=allow_unknown_tests,
    )
    try:
        runtime = RUNTIME_REGEX.search(result.stdout)
        runtime = float(runtime.group(1))
    except (AttributeError, TypeError, ValueError):
        runtime = result.runtime
    try:
        memory_used = MEMORY_REGEX.search(result.stdout)
        memory_used = int(memory_used.group(1))
    except (AttributeError, TypeError, ValueError):
        memory_used = float("inf")
        if outcome != PredictionOutcome.COMPILE_ERROR and expecting_memory:
            outcome = PredictionOutcome.MISSING_INFO
    return outcome, parsed_test_cases, memory_used, runtime
