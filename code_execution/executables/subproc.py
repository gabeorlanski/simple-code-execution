"""This module contains the SubprocessExecutable class, which is used to execute commands using subprocesses."""

import asyncio
import pathlib
import subprocess
import tempfile
import time
from typing import Dict, List, Literal, Optional

import aiofiles
import structlog
from pydantic import BaseModel

from code_execution.executables import base

logger = structlog.get_logger()


class SubprocessResult(base.BaseResult):
    """The result of execution using subprocesses."""

    stderr: str


class SubprocessExecutable(base.BaseExecutable):
    """The executable for running commands using subprocesses."""


class SubprocessExecutableResult(base.ExecutableResult):
    """The result of executing a subprocess executable."""

    executable_type: Literal["subprocess"] = "subprocess"


def _execute(
    command_to_run: List[str],
    working_dir: pathlib.Path,
    timeout: int,
    stdin: Optional[str | List[str]] = None,
) -> SubprocessResult:
    """Executes a single command."""
    timed_out = False
    return_code = -1
    runtime = timeout
    stderr = None
    stdout = None
    had_unexpected_error = False
    start_time = time.time()
    execution_process = subprocess.Popen(
        command_to_run,
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    if stdin:
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        stdin = stdin.encode("utf-8")
    else:
        stdin = None
    try:
        try:
            outputs = execution_process.communicate(
                input=stdin, timeout=timeout
            )
            t1 = time.time()
            stdout = outputs[0].decode("utf-8")
            stderr = outputs[1].decode("utf-8")
            runtime = t1 - start_time
            return_code = execution_process.returncode

        except subprocess.TimeoutExpired:
            stdout = stderr = ""
            runtime = timeout
            return_code = 0
            timed_out = True
        execution_process.kill()

    # pylint: disable=broad-except
    except Exception as e:
        stderr = str(e)
        stdout = ""
        return_code = -1
        runtime = -1
        timed_out = False
        had_unexpected_error = True
        execution_process.kill()
    return SubprocessResult(
        return_code=return_code,
        runtime=runtime,
        stderr=stderr,
        output=stdout,
        timed_out=timed_out,
        had_unexpected_error=had_unexpected_error,
    )


async def _write_files_async(
    temp_dir: pathlib.Path, files: dict[str, str]
) -> None:
    """Writes files asynchronously to the temporary directory."""

    async def write_single_file(file_name: str, file_content: str) -> None:
        logger.debug("Writing file: %s", file_name)
        async with aiofiles.open(temp_dir / file_name, "w") as f:
            await f.write(file_content)

    await asyncio.gather(
        *[
            write_single_file(file_name, file_content)
            for file_name, file_content in files.items()
        ]
    )


@base.RunnerRegistry.register("subprocess")
async def execute_subprocess_async(
    executable: SubprocessExecutable,
) -> SubprocessExecutableResult:
    """Executes the subprocess executable."""
    logger.info("Executing subprocess executable: %s", repr(executable))
    results = []
    with tempfile.TemporaryDirectory(prefix="subproc_execution_") as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        # Write files asynchronously but wait for completion before proceeding
        await _write_files_async(temp_dir, executable.files)

        t0 = time.time()
        for command in executable.commands:
            logger.debug("Executing command: %s", command.command)
            res = _execute(
                command.command, temp_dir, command.timeout, stdin=command.stdin
            )
            results.append(res)
            if executable.early_stopping and res.had_error:
                logger.debug("Early stopping due to error")
                break
        t1 = time.time()
        logger.debug("Execution time: %s", t1 - t0)
        tracked_files = {}
        for tracked_file in executable.tracked_files:
            logger.debug("Reading tracked file: %s", tracked_file)
            tracked_files[tracked_file] = (temp_dir / tracked_file).read_text()
    return SubprocessExecutableResult(
        results=results,
        elapsed=t1 - t0,
        tracked_files=tracked_files,
    )


def execute_subprocess(
    executable: SubprocessExecutable,
) -> SubprocessExecutableResult:
    """Synchronous wrapper for execute_subprocess."""
    return asyncio.run(execute_subprocess_async(executable))
