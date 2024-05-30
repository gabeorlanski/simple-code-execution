from copy import deepcopy

import pytest

from code_execution import code_trees


def test_remove_deep_code():
    long_codes = []

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
