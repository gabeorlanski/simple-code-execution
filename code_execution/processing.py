"""
Functions for preprocessing and postprocessing the commands to run."""

"""Module for entrypoints for code execution."""

import dataclasses
import inspect
import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from code_execution.configs import ExecutionConfig
from code_execution.data_structures import Command
from code_execution.data_structures import CommandsToRun
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.utils import get_pred_dir
from code_execution.utils import run_in_parallel
from code_execution.utils import wrap_processor

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PredTimingsCollection:
    """Stores the timings per prediction for different aspects."""

    preprocess_time: Dict[str, float]
    writing_time: Dict[str, float]
    cleanup_time: Dict[str, float]

    def __getitem__(self, key):
        return {
            "preprocess_time": self.preprocess_time.get(key, 0.0),
            "writing_time": self.writing_time.get(key, 0.0),
            "cleanup_time": self.cleanup_time.get(key, 0.0),
        }


def default_postprocessor(
    prediction: Dict, result: ExecutionResult, **_
) -> Dict:
    """Adds the result to the prediction dict."""
    return {**prediction, **result.to_dict()}


def preprocess_commands(
    config: ExecutionConfig,
    dir_to_use: Path,
    pred_list: List[Dict],
    preprocessor: Callable[[Dict], Union[Executable, ExecutionResult]],
    preproc_returns_list: bool = False,
    batch_size: int = 1,
) -> Tuple[List[Dict], List[Dict], Dict[Tuple[int, int], ExecutionResult]]:
    """Preprocesses the commands to run.
    Args:
        config: The execution config.
        dir_to_use: The directory to use for execution.
        pred_list: The list of predictions.
        preprocessor: The preprocessor to use.
        preproc_returns_list: Whether the preprocessor returns a list of executables.
        batch_size: The batch size to use for execution.
        error_directory: The directory to save errors to.
    Returns:
        files_to_write: The files to write to disk.
        commands_to_run: The commands to run.
        filtered_out: The results that were filtered out during preprocessing,
            these will be added back after execution.
        timings: The timings of the preprocessing each example.
    """
    logger.debug("Creating Executables")
    executable_creator = wrap_processor(
        preprocessor,
        batch_size=batch_size,
        returns_list=preproc_returns_list,
    )
    if batch_size > 1:
        logger.debug(
            "Chunking %s predictions into batches of %d",
            f"{len(pred_list):,}",
            batch_size,
        )
        process_args = []
        current_batch = []
        for idx, pred in enumerate(pred_list):
            current_batch.append({"idx": idx, "args": [pred]})
            if len(current_batch) == batch_size:
                process_args.append(current_batch)
                current_batch = []
        if current_batch:
            process_args.append(current_batch)
    else:
        process_args = [
            {"idx": idx, "args": [pred]} for idx, pred in enumerate(pred_list)
        ]
    logger.debug("Processing %d batche(s)", len(process_args))
    results = run_in_parallel(
        executable_creator,
        process_args,
        desc="Processing Code",
        num_workers=config.num_workers,
        max_tasks_per_process=config.max_tasks_per_process,
        disable_tqdm=config.disable_tqdm,
        chunk_size=config.chunk_size,
        target_returns_multiple=preproc_returns_list or batch_size > 1,
    )

    commands_to_run = []
    files_to_write = []
    filtered_out = {}
    timings = {}

    for (idx, sub_idx), preproc_time, exec_command in results:
        idx_use = f"{idx}.{sub_idx}"
        timings[idx_use] = preproc_time
        if isinstance(exec_command, ExecutionResult):
            filtered_out[(idx, sub_idx)] = dataclasses.replace(
                exec_command, key=(idx, sub_idx), preprocess_time=preproc_time
            )

            continue

        pred_dir = get_pred_dir(idx_use, dir_to_use)
        files_to_write.append((idx_use, exec_command.files, pred_dir))
        cmds = []
        for command in exec_command.commands:
            if not isinstance(command, Command):
                command = Command(**command)

            if command.timeout is None:
                command.timeout = config.default_timeout
            cmds.append(command)
        commands_to_run.append(
            {
                "key": (
                    idx,
                    sub_idx,
                ),  # (idx, sub_idx) is the key for the result
                "executable": CommandsToRun(
                    cwd=pred_dir.resolve().absolute(),
                    commands=cmds,
                    tracked_files=exec_command.tracked_files,
                    should_early_stop=exec_command.should_early_stop,
                    stdout_postprocessor=exec_command.stdout_postprocessor,
                ),
            }
        )
    logger.info(f"{len(commands_to_run):,} commands to run")
    logger.info(f"{len(filtered_out):,} were filtered out.")
    return files_to_write, commands_to_run, filtered_out, timings


def postprocess_commands(
    raw_preds: Dict,
    results: Dict[Tuple[int, int], ExecutionResult],
    postprocessor: Callable[[Dict, ExecutionResult], Dict],
    timings: PredTimingsCollection,
    returned_multiple: bool,
    disable_tqdm: bool = False,
    log_freq: int = 1000,
) -> List[Dict]:
    """Postprocesses the commands after exeuction.

    Args:
        raw_preds (Dict): The raw predictions before postprocessing, used to add
            back information.
        results (Dict[Tuple[int, int], ExecutionResult]): The results of
            executions where the key is used for ordering and the value is the
            result post execution.
        postprocessor (Callable): The postprocessor function to use.
        timings (PredTimingsCollection): The timings of the predictions.
        returned_multiple (bool): Whether the preprocessor returned multiple
            results per prediction.
        disable_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        log_freq (int, optional): How often to log. Defaults to 1000.
    Returns:
        List[Dict]: The postprocessed results.
    """
    logger.debug("Postprocessing %d predictions", len(results))

    # Add the timings to the results
    for key, result in results.items():
        results[key] = dataclasses.replace(
            result, **timings[".".join(map(str, key))]
        )

    if returned_multiple:
        logger.info("Multiple results per prediction, grouping them")
        new_results = defaultdict(list)
        for key, result in sorted(results.items(), key=lambda x: x[0]):

            new_results[(key[0],)].append(result)

        results = new_results

    out = []
    if disable_tqdm:
        prog_level = logging.INFO
        res_generator = sorted(results.items(), key=lambda x: x[0])
    else:
        prog_level = logging.DEBUG
        res_generator = tqdm(
            sorted(results.items(), key=lambda x: x[0]),
            desc="Postprocessing",
            total=len(results),
            mininterval=log_freq,
        )

    for key, result in res_generator:

        prediction = raw_preds[key[0]]

        start = datetime.now()
        processed = postprocessor(prediction, result)
        elapsed = (datetime.now() - start).total_seconds()
        processed["postprocess_time"] = elapsed
        out.append(processed)
        if len(out) % log_freq == 0:
            logger.log(
                prog_level,
                f"Processed {len(out):,}/{len(results):,} predictions",
            )
    return out
