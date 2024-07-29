from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@pytest.fixture()
def pass_print():
    yield "PASSED"


@pytest.fixture()
def fail_print():
    yield "FAILED"


@pytest.fixture()
def passing_program(pass_print):
    yield (
        "from collections import Counter\n"
        "def f():\n   return Counter([1, 2, 3])\n"
        "assert f() == {1: 1, 2: 1, 3: 1}\n"
        f"print('{pass_print}')"
    )


@pytest.fixture()
def error_program(fail_print):
    yield (
        "from collections import Counter\n"
        "def f():\n   return Counter([1, 2, 3])\n"
        f"print('{fail_print}')\n"
        f"raise ValueError('{fail_print}')"
    )


@pytest.fixture()
def timeout_program(pass_print):
    yield f"from time import sleep\nprint('{pass_print}')\nsleep(10)"


@pytest.fixture()
def stdin_program():
    yield "print('Input 1: ' + input())\nprint('Input 2: ' + input())"


@pytest.fixture()
def loop_stdin_program():
    yield "while True:\n    print('Input: ' + input())"
