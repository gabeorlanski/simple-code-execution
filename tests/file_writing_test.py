from pathlib import Path
from unittest import mock

import pytest

from code_execution import file_writing


@pytest.fixture()
def dummy_files(tmpdir):
    files = []
    expected_files = []
    tmpdir = Path(tmpdir)
    for i in range(10):
        files.append(
            (
                i,
                {
                    "main.py": f"test_{i}",
                },
                tmpdir / f"pred{i}",
            )
        )
        expected_files.append((f"test_{i}", tmpdir / f"pred{i}" / "main.py"))

    yield expected_files, files


@pytest.mark.parametrize(
    "in_notebook", [True, False], ids=["notebook", "not_notebook"]
)
def test_write_files(in_notebook, dummy_files):
    expected, files = dummy_files
    with mock.patch(
        "code_execution.file_writing.utils.in_notebook",
        return_value=in_notebook,
    ):
        file_writing.write_executables(files, 10)

    for expected_content, expected_path in expected:
        assert expected_path.exists()
        with open(expected_path) as f:
            assert f.read() == expected_content
