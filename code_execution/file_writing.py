""" Module for handling writing executables to disk. """

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import aiofiles
from tqdm.asyncio import tqdm_asyncio

from code_execution import utils

logger = logging.getLogger(__name__)


async def _async_write_executables(
    pred_list: List[Dict],
    rate_limit: int,
    enable_tqdm: bool,
):
    """Writes the executables to the disk asynchronously."""
    sem = asyncio.Semaphore(rate_limit)

    async def write_pred(idx, files, pred_dir):
        async with sem:
            pred_dir.mkdir(exist_ok=True)

            for name, contents in files.items():
                filepath = pred_dir.joinpath(name)
                async with aiofiles.open(
                    str(filepath), "w", encoding="utf-8"
                ) as f:
                    await f.write(contents)
            return idx, pred_dir.resolve().absolute()

    tasks = [write_pred(*p) for p in pred_list]
    if enable_tqdm:
        gen = tqdm_asyncio.as_completed(tasks, desc="Writing Executables")
    else:
        gen = asyncio.as_completed(tasks)
    out = []
    for result in gen:
        res = await result
        out.append(res)
        if len(out) % 100_000 == 0:
            logger.info(f"Wrote {len(out)}/{len(pred_list)} predictions")

    return out


def write_executables(
    files_to_write: List[Tuple],
    write_rate_limit: int,
    enable_tqdm: bool = False,
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
    )
    logger.debug("Ensuring all files written...")
    for idx, r in out_results:
        if not r.exists():
            raise ValueError(f"Directory for {idx} does not exist at {r}")


async def _async_cleanup(
    pred_list: List[Tuple],
    rate_limit: int,
    enable_tqdm: bool = False,
):
    """Cleans up the executables on the disk asynchronously."""
    sem = asyncio.Semaphore(rate_limit)

    async def cleanup_dir(pred_dir: Path):
        async with sem:
            shutil.rmtree(pred_dir)

    tasks = [cleanup_dir(p[-1]) for p in pred_list]
    if not enable_tqdm:
        gen = asyncio.as_completed(tasks)
    else:
        gen = tqdm_asyncio.as_completed(tasks, desc="Cleaning Up")
    completed = 0
    for result in gen:
        _ = await result
        completed += 1
        if completed % 100_000 == 0:
            logger.info(f"Deleted {completed}/{len(pred_list)} predictions")


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

    utils.notebook_safe_async_run(
        _async_cleanup,
        pred_list=files,
        rate_limit=rate_limit,
        enable_tqdm=enable_tqdm,
    )

    logger.debug("Ensuring all files cleaned up...")
    for *_, d in files:
        if d.exists():
            raise ValueError(f"Directory for {d} exists")
