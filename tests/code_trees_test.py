from copy import deepcopy

import pytest

from code_execution import code_trees
import ast


def test_remove_deep_code():
    def make_deep_code(d):
        if d == 0:
            return f"({d},-1)"
        return f"({d},{make_deep_code(d-1)})"

    code_lines = [
        [
            make_deep_code(500),
            make_deep_code(99),
            "def f(x):\n\treturn x",
        ],
        [
            make_deep_code(499),
            make_deep_code(7),
            "def f(x):\n\treturn x",
        ],
        [
            make_deep_code(498),
            make_deep_code(5),
            "def f(x):\n\treturn x",
        ],
        [
            make_deep_code(497),
            make_deep_code(3),
            "def f(x):\n\treturn x",
        ],
        [
            make_deep_code(496),
            make_deep_code(1),
            "def f(x):\n\treturn x",
        ],
    ]

    res = code_trees.remove_trees_from_lists(deepcopy(code_lines), timeout=0.1)
    assert res == [c[1:] for c in code_lines]


@pytest.mark.parametrize(
    "test_list,expected",
    [
        [("f(x)", "1", False), "assert f(x) == 1"],
        [("f(x)", "1.0", True), "assert f(x) - 1.0 < 1e-06"],
    ],
    ids=["simple", "float"],
)
@pytest.mark.parametrize("return_str", [True, False], ids=["str", "tree"])
def test_convert_call_to_assert(test_list, expected, return_str):
    result = code_trees.convert_call_to_assert(
        *test_list, return_str=return_str
    )
    if not return_str:
        result = ast.unparse(result)
    assert result == expected
