import pytest


@pytest.fixture()
def basic_prediction():
    test_cases = [
        ("add_1(1)", "2", False),
        ("add_1(-1)", "0", False),
        ("add_1(0)", "1", False),
        ("add_1(5)", "5", False),
        ("add_1('s')", "5", False),
    ]

    code = """def add_1(x):\n   return x + 1"""
    entry_point = "add_1"
    yield {
        "solution": code,
        "entry_point": entry_point,
        "test_cases": test_cases,
    }


@pytest.fixture()
def uncovered_prediction():
    code = """def uncovered(x,y):
    if x == 1 and y == 0:
        return True
    if x == 0 and y == 1:
        return True
    if x == 1 and y == 1:
        return False
    return True
    """

    test_cases = [
        ("uncovered(0,1)", "True", False),
        ("uncovered(1,0)", "True", False),
        ("uncovered(0,0)", "True", False),
    ]
    yield {
        "solution": code,
        "entry_point": "uncovered",
        "test_cases": test_cases,
    }
