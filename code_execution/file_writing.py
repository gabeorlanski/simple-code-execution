"""Module for handling writing executables to disk."""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import aiofiles
from tqdm.asyncio import tqdm_asyncio

from code_execution import utils

logger = logging.getLogger(__name__)


class WritingFailure(Exception):
    """Exception raised when writing a file fails."""

    pass


async def _async_write_executables(
    pred_list: List[Dict],
    rate_limit: int,
    enable_tqdm: bool,
    log_freq: int,
):
    """Writes the executables to the disk asynchronously."""
    sem = asyncio.Semaphore(rate_limit)

    async def write_pred(idx, files, pred_dir):
        async with sem:
            pred_dir.mkdir(exist_ok=True)
            start = datetime.now()
            for name, contents in files.items():
                filepath = pred_dir.joinpath(name)
                try:
                    async with aiofiles.open(
                        str(filepath), "w", encoding="utf-8"
                    ) as f:
                        await f.write(contents)
                except Exception as e:
                    logger.error(
                        "Error writing file %s to %s: %s",
                        name,
                        pred_dir,
                        e,
                    )
                    logger.error("Contents: %s", contents)
                    raise WritingFailure(
                        f"Failed to write {name} to {pred_dir} because of {e}"
                    ) from e
            write_elapsed = (datetime.now() - start).total_seconds()
            return idx, write_elapsed, pred_dir.resolve().absolute()

    tasks = [write_pred(*p) for p in pred_list]
    if enable_tqdm:
        gen = tqdm_asyncio.as_completed(tasks, desc="Writing Executables")
    else:
        gen = asyncio.as_completed(tasks)
    out = []
    for result in gen:
        res = await result
        out.append(res)
        if len(out) % log_freq == 0:
            logger.info(f"Wrote {len(out)}/{len(pred_list)} predictions")

    return out


def write_executables(
    files_to_write: List[Tuple],
    write_rate_limit: int,
    enable_tqdm: bool = False,
    log_freq: int = 100_000,
):
    """Writes the executables to the disk.

    Args:
        files_to_write (List[Dict]): The list of files to write. Each item is a
            dict where the key is a absolute path to the file and the value is
            the contents.
        write_rate_limit (int): The asynchronous write rate limit.
        enable_tqdm (bool, optional): Whether to enable the progress bars. Defaults to False.

    Raises:
        ValueError: If the prediction directory does not exist.
    """

    logger.debug(
        "Writing %d file(s) to disk with %s",
        len(files_to_write),
        f"{write_rate_limit=}",
    )
    out_results = utils.notebook_safe_async_run(
        _async_write_executables,
        pred_list=files_to_write,
        rate_limit=write_rate_limit,
        enable_tqdm=enable_tqdm,
        log_freq=log_freq,
    )
    logger.debug("Ensuring all files written...")
    times = {}
    for idx, write_time, r in out_results:
        times[idx] = write_time
        if not r.exists():
            raise ValueError(f"Directory for {idx} does not exist at {r}")
    logger.info("Wrote all files to disk")
    return times


async def _async_cleanup(
    pred_list: List[Tuple],
    rate_limit: int,
    enable_tqdm: bool = False,
):
    """Cleans up the executables on the disk asynchronously."""
    sem = asyncio.Semaphore(rate_limit)

    async def cleanup_dir(pidx, pred_dir: Path):
        async with sem:
            start = datetime.now()
            shutil.rmtree(pred_dir)
            return pidx, (datetime.now() - start).total_seconds()

    tasks = [cleanup_dir(p[0], p[-1]) for p in pred_list]
    if not enable_tqdm:
        gen = asyncio.as_completed(tasks)
    else:
        gen = tqdm_asyncio.as_completed(tasks, desc="Cleaning Up")
    completed = 0
    times = {}
    for result in gen:
        idx, elapsed = await result
        times[idx] = elapsed
        completed += 1
        if completed % 100_000 == 0:
            logger.info(f"Deleted {completed}/{len(pred_list)} predictions")
    return times


def cleanup(
    files: List[Tuple],
    rate_limit: int,
    enable_tqdm: bool = False,
):
    """Cleans up the executables on the disk.

    Args:
        files (List[Tuple]): The list of files to clean up.
        rate_limit (int): The rate limit (# threads) for cleaning up the files.
        disable_tqdm (bool): Disable the progress bars.
        quiet (bool, optional): Whether to suppress logging. Defaults to False.
    Raises:
        ValueError: If the prediction directory exists after cleanup.
    """
    logger.debug(
        "Cleaning up %d predictions with rate limit of %s",
        len(files),
        rate_limit,
    )

    times = utils.notebook_safe_async_run(
        _async_cleanup,
        pred_list=files,
        rate_limit=rate_limit,
        enable_tqdm=enable_tqdm,
    )

    logger.debug("Ensuring all files cleaned up...")
    for *_, d in files:
        if d.exists():
            raise ValueError(f"Directory for {d} exists")
    return times
