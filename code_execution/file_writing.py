""" Module for handling writing executables to disk. """

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import aiofiles
from tqdm.asyncio import tqdm_asyncio
from code_execution.utils import run_in_parallel

logger = logging.getLogger(__name__)


async def _async_write_executables(
    pred_list: List[Dict],
    rate_limit: int,
    disable_tqdm: bool,
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
    if disable_tqdm:
        gen = asyncio.as_completed(tasks)
    else:
        gen = tqdm_asyncio.as_completed(tasks, desc="Writing Executables")
    out = []
    for result in gen:
        res = await result
        out.append(res)
        if disable_tqdm and len(out) % 500 == 0:
            print(f"Wrote {len(out)}/{len(pred_list)} predictions")

    return out


def _write_worker(batch):
    out = []
    for idx, files, pred_dir in batch:
        pred_dir.mkdir(exist_ok=True)
        for name, contents in files.items():
            filepath = pred_dir.joinpath(name)
            with filepath.open("w", encoding="utf-8") as f:
                f.write(contents)
        out.append((idx, pred_dir.resolve().absolute()))
    return out


def write_executables(
    files_to_write: List[Tuple],
    write_rate_limit: int,
    disable_tqdm: bool,
    use_mp: bool = False,
):
    """Writes the executables to the disk.

    Args:
        files_to_write (List[Dict]): The list of files to write. Each item is a
            dict where the key is a absolute path to the file and the value is
            the contents.
        write_rate_limit (int): The asynchronous write rate limit.
        disable_tqdm (bool): Disable the progress bars.

    Raises:
        ValueError: If the prediction directory does not exist.
    """

    logger.debug(
        "Writing %d file(s) to disk with %s",
        len(files_to_write),
        f"{write_rate_limit=}",
    )
    if use_mp:
        out_results = run_in_parallel(
            _write_worker,
            [
                files_to_write[i : i + 100]
                for i in range(0, len(files_to_write), 100)
            ],
            num_workers=min(write_rate_limit, 8),
            desc="Writing Executables",
            target_returns_multiple=True,
        )
    else:

        out_results = asyncio.run(
            _async_write_executables(
                files_to_write,
                rate_limit=write_rate_limit,
                disable_tqdm=disable_tqdm,
            )
        )
    logger.debug("Ensuring all files written...")
    for idx, r in out_results:
        if not r.exists():
            raise ValueError(f"Directory for {idx} does not exist at {r}")


async def _async_cleanup(
    pred_list: List[Tuple],
    rate_limit: int,
    disable_tqdm: bool,
):
    """Cleans up the executables on the disk asynchronously."""
    sem = asyncio.Semaphore(rate_limit)

    async def cleanup_dir(pred_dir: Path):
        async with sem:
            shutil.rmtree(pred_dir)

    tasks = [cleanup_dir(p[-1]) for p in pred_list]
    if disable_tqdm:
        gen = asyncio.as_completed(tasks)
    else:
        gen = tqdm_asyncio.as_completed(tasks, desc="Cleaning Up")
    completed = 0
    for result in gen:
        _ = await result
        completed += 1
        if disable_tqdm and completed % 500 == 0:
            print(f"Wrote {completed}/{len(pred_list)} predictions")


def _cleanup_worker(batch):
    for *_, d in batch:
        shutil.rmtree(d)
    return [1 for _ in batch]


def cleanup(
    files: List[Tuple],
    rate_limit: int,
    disable_tqdm: bool,
    use_mp: bool = False,
):
    """Cleans up the executables on the disk.

    Args:
        files (List[Tuple]): The list of files to clean up.
        rate_limit (int): The rate limit (# threads) for cleaning up the files.
        disable_tqdm (bool): Disable the progress bars.

    Raises:
        ValueError: If the prediction directory exists after cleanup.
    """
    logger.debug(
        "Cleaning up %d predictions with rate limit of %s",
        len(files),
        rate_limit,
    )
    if use_mp:
        _ = run_in_parallel(
            _cleanup_worker,
            [files[i : i + 100] for i in range(0, len(files), 100)],
            num_workers=min(rate_limit, 8),
            desc="Cleaning Up",
            target_returns_multiple=True,
        )
    else:
        asyncio.run(
            _async_cleanup(
                files,
                rate_limit=rate_limit,
                disable_tqdm=disable_tqdm,
            )
        )

    logger.debug("Ensuring all files cleaned up...")
    for *_, d in files:
        if d.exists():
            raise ValueError(f"Directory for {d} exists")
