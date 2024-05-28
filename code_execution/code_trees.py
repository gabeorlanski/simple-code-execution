import ast
import logging
from typing import Callable, List, Tuple, Union

from .utils import ContextTimeLimitException
from .utils import time_limit

logger = logging.getLogger(__name__)


def safe_ast_parse(code) -> ast.Module:
    """Safely parse a string of code into an AST, if possible. Otherwise return None."""
    try:
        with time_limit(5):
            res = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, ContextTimeLimitException):
        return None
    return res


def is_valid_python(code):
    """Checks if the code is valid python."""
    return safe_ast_parse(code) is not None


def is_simple_test_case(tree):
    """Checks if the test case is an assert with a function call on the left."""
    if not tree.body:
        return False

    n = tree.body[0]
    if not isinstance(n, ast.Assert):
        return False
    if not isinstance(n.test, ast.Compare):
        return False
    left = n.test.left
    for c in ast.walk(left):
        if isinstance(c, ast.Call):
            return True
    return False


def get_global_imports(tree: ast.Module) -> List[str]:
    """ Get the global imports from an ast tree as a list of strings."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            out.append(ast.unparse(node))
    return out


def convert_call_to_assert(
    call: str,
    expected_output: str,
    requires_float=False,
    return_str: bool = False,
) -> Union[ast.Module, str]:
    """Coverts call code to an assertion with an expected output.

    The call code must end in an ast.Expr node, which is the node that will be
    converted to an assertion.

    The expected output must be an expression.

    Args:
        call: The code to be converted to an assertion.
        expected_output: The expected output of the call.
        requires_float: Whether the expected output is a float. If so, we will
            add a tolerance of 1e-6.
        return_str: Whether to return the converted code as a string or as an ast tree.

    Returns:
        The converted ast tree or the converted code.

    """
    tree = ast.parse(call)
    if isinstance(tree.body[-1], ast.Assert):
        return ast.unparse(tree) if return_str else tree
    out_tree = ast.parse(expected_output).body[0].value
    if requires_float:
        tree.body[-1] = ast.Assert(
            test=ast.Compare(
                left=ast.BinOp(
                    left=tree.body[-1].value, op=ast.Sub(), right=out_tree
                ),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value="1e-6", kind=None)],
            ),
            msg=None,
        )
    else:
        tree.body[-1] = ast.Assert(
            test=ast.Compare(
                left=tree.body[-1].value, ops=[ast.Eq()], comparators=[out_tree]
            ),
            msg=None,
        )

    if return_str:
        return ast.unparse(tree)

    return tree


def convert_test_list_to_assert(
    test_list: List[Union[Tuple[str,str,bool],str]], timeout: float = -1.0, convert_to_string: bool = False
) -> List[Union[ast.AST, str]]:
    """ Converts a list of test cases to assertion nodes.
    
    Args:
        test_list: A list of test cases. Each test case can be a string or a tuple
            of (call, output, requires_float). If the test case is a string, it will
            be parsed as a call. If it is a tuple, it will be converted to an assertion.
        timeout: The timeout for parsing the test cases.
        convert_to_string: Whether to convert the resulting AST to a string.
    
    Returns:
        A list of converted test cases as AST nodes or strings.
    """
    out = []
    for tc in test_list:
        try:
            if isinstance(tc, str):
                with time_limit(timeout):
                    tree = safe_ast_parse(tc)
            else:
                i, o, *rest = tc
                requires_float = False
                if len(rest) > 0:
                    requires_float = rest[0]
                with time_limit(timeout):
                    tree = convert_call_to_assert(
                        i, o, requires_float=requires_float
                    )
        except ContextTimeLimitException:
            tree = None
        if tree is not None:
            if convert_to_string:
                out.append(ast.unparse(tree))
            else:
                out.append(tree)
    return out


def wrap_assert_in_try_print(
    idx: int,
    call: str,
    output: str,
    requires_float: bool,
    print_formatter: Callable[[int], Tuple[str, str, List[Tuple[str, str]]]],
) -> str:
    """Wraps a test case in a try-except block that prints the result.

    The resulting code will be:
    ```
    try:
        {ASSERTION}
        print({pass_str})
    except AssertionError:
        print({fail_str})
    ```
    The exceptions will be appended as:
    ```
    except {exception_type} as e:
        print({print_string})
    ```

    Args:
        idx: The index of the test case.
        call: The call code.
        output: The expected output code.
        requires_float: Whether the expected output is a float.
        print_formatter: A function that takes in the index and returns the
            pass, fail, and a list of length 2 tuples for exceptions. For the
            exception strings, they should be in the format (exception_type,
            print_string). The except clause will be
            `except {exception_type} as e\n\tprint({print_string}). The
            resulting strings will be passed directly to print().
    """
    tree = convert_call_to_assert(
        call=call, expected_output=output, requires_float=requires_float
    )
    pass_str, fail_str, error_strs = print_formatter(idx)
    template = (
        f"try:\n\tprint({pass_str})\n"
        f"except AssertionError:\n\tprint({fail_str})\n"
    )
    for e_type, e_str in error_strs:
        template += f"except {e_type} as e:\n\tprint({e_str})\n"

    template_tree = ast.parse(template).body[0]
    template_tree.body = tree.body + template_tree.body
    return ast.unparse(ast.fix_missing_locations(template_tree))
