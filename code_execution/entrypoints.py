""" Module for entrypoints for code execution. """

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from code_execution.configs import ExecutionConfig
from code_execution.data_structures import Command
from code_execution.data_structures import CommandsToRun
from code_execution.data_structures import Executable
from code_execution.execution import CommandResult
from code_execution.execution import ExecutionResult
from code_execution.execution import execute_commands
from code_execution.file_writing import cleanup
from code_execution.file_writing import write_executables
from code_execution.processors.utils import ParsedTestResults
from code_execution.processors.utils import PredictionOutcome
from code_execution.utils import get_pred_dir
from code_execution.utils import run_in_parallel
from code_execution.utils import wrap_processor

logger = logging.getLogger(__name__)


def sanitize_execution_result(result):
    """Sanitizes the execution result for JSON serialization."""
    if isinstance(result, PredictionOutcome):
        return result.name
    if isinstance(result, ExecutionResult):
        return result.to_dict()
    if isinstance(result, (CommandResult, ParsedTestResults)):
        return asdict(result)
    if isinstance(result, tuple):
        return tuple(sanitize_execution_result(r) for r in result)
    if isinstance(result, list):
        return [sanitize_execution_result(r) for r in result]
    if isinstance(result, dict):
        return {k: sanitize_execution_result(v) for k, v in result.items()}
    return result


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
    for (idx, sub_idx), exec_command in results:
        idx_use = f"{idx}.{sub_idx}"
        if isinstance(exec_command, ExecutionResult):
            filtered_out[(idx, sub_idx)] = exec_command
            continue

        pred_dir = get_pred_dir(idx_use, dir_to_use)
        files_to_write.append((idx_use, exec_command.files, pred_dir))
        cmds = []
        for command in exec_command.commands:
            # TODO(gabeorlanski): Remove this eventually when all preprocessors return Commands
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
                    ensure_all_run=exec_command.ensure_all_run,
                    should_early_stop=exec_command.should_early_stop,
                ),
            }
        )
    logger.info(f"{len(commands_to_run):,} commands to run")
    logger.info(f"{len(filtered_out):,} were filtered out.")
    return files_to_write, commands_to_run, filtered_out


def postprocess_commands(
    raw_preds: Dict,
    results: Dict[Tuple[int, int], ExecutionResult],
    postprocessor: Callable[[Dict, ExecutionResult], Dict],
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
        returned_multiple (bool): Whether the preprocessor returned multiple
            results per prediction.
        disable_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        log_freq (int, optional): How often to log. Defaults to 1000.
    Returns:
        List[Dict]: The postprocessed results.
    """
    logger.debug("Postprocessing %d predictions", len(results))
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
        processed = postprocessor(prediction, result)
        out.append(processed)
        if len(out) % log_freq == 0:
            logger.log(
                prog_level,
                f"Processed {len(out):,}/{len(results):,} predictions",
            )
    return out


def _write_maybe_save_error_dir(
    config, files, raw_preds, exec_dir: Path, error_directory: Optional[Path]
):
    try:
        write_executables(
            files_to_write=files,
            write_rate_limit=config.write_rate_limit,
            enable_tqdm=config.display_write_progress,
            log_freq=config.write_log_freq,
        )
    except Exception as e:
        logger.exception("Error writing executables")

        if error_directory:
            error_directory.mkdir(parents=True, exist_ok=True)
            with error_directory.joinpath("exec_files.txt").open("w") as dir_f:
                for f in os.listdir(exec_dir.absolute()):
                    dir_f.write(f"{exec_dir/f}\n")

            error_file = error_directory.joinpath("errors.jsonl")
            with error_file.open("w") as error_f:
                for idx, files, pred_dir in files:
                    if isinstance(idx, str):
                        use_idx = int(idx.split(".")[0])
                    elif isinstance(idx, tuple):
                        use_idx = idx[0]
                        idx = f"{idx[0]}.{idx[1]}"
                    else:
                        use_idx = int(idx)

                    try:
                        to_write = json.dumps(
                            {
                                "use_idx": idx,
                                "pred": raw_preds[use_idx],
                                "files": files,
                                "pred_dir": str(pred_dir),
                            }
                        )
                    except json.JSONDecodeError:
                        to_write = json.dumps(
                            {
                                "use_idx": idx,
                                "files": "Error decoding files",
                                "pred_dir": str(pred_dir),
                            }
                        )

                    error_f.write(to_write + "\n")
        raise e


def execute_predictions(
    config: ExecutionConfig,
    pred_list: List[Dict],
    preprocessor: Callable[[Dict], Union[Executable, List[Executable]]],
    postprocessor: Callable[[Dict, Dict], Dict] = None,
    debug_dir: Path = None,
    preproc_returns_list: bool = False,
    preproc_batch_size: int = 1,
    error_directory: Optional[Path] = None,
) -> List[Dict]:
    """Executes the program predictions.

    First preprocesses the commands to run, writes them to disk, then executes
    them, and finally postprocesses the results.

    Args:
        config (ExecutionConfig): The config for execution.
        pred_list (List[Dict]): The list of predictions to execute.
        preprocessor (Callable[[Dict], Union[Executable, List[Executable]]]): The preprocessor
            function to use to create the files and commands to execute.
        postprocessor (Callable[[Dict, Dict], Dict], optional): The postprocessor
            function for taking the results and processing them. Defaults to None.
        debug_dir (Path, optional): The directory to use if you want to debug. This saves
            **all** files here and does not clean them up. Defaults to None.
        preproc_returns_list (bool, optional): Is the preprocess function one-to-one
            or one-to-many. Defaults to False.
        preproc_batch_size (int, optional): The batch size for preprocessing. Defaults to 1.
        error_directory (Path, optional): The directory to save errors to. Defaults to None.
    Returns:
        List[Dict]: The executed predictions.
    """
    if postprocessor is None:
        logger.info("Using default postprocessor")
        postprocessor = default_postprocessor
    logger.debug(
        "Starting execution with %s predictions", f"{len(pred_list):,}"
    )

    def _run(dir_to_use):
        logger.debug("Using %s as execution directory", dir_to_use)
        logger.debug("Preprocessing commands")
        files_to_write, commands_to_run, filtered_results = preprocess_commands(
            config=config,
            dir_to_use=dir_to_use,
            pred_list=pred_list,
            preprocessor=preprocessor,
            preproc_returns_list=preproc_returns_list,
            batch_size=preproc_batch_size,
        )

        file_chunks = []
        command_chunks = []
        if (
            config.max_execute_at_once > 0
            and len(commands_to_run) > config.max_execute_at_once
        ):
            logger.info(
                f"Executing {len(commands_to_run):,} commands"
                f"in chunks of {config.max_execute_at_once:,}"
            )
            if config.max_execute_at_once == 1:
                logger.warning(
                    "max_execute_at_once is set to 1, this will be slow."
                )
                file_chunks = [[f] for f in files_to_write]
                command_chunks = [[c] for c in commands_to_run]
            else:
                for i in range(
                    0, len(commands_to_run), config.max_execute_at_once
                ):
                    file_chunks.append(
                        files_to_write[i : i + config.max_execute_at_once]
                    )
                    command_chunks.append(
                        commands_to_run[i : i + config.max_execute_at_once]
                    )
        else:
            logger.info(f"Executing {len(commands_to_run):,} commands")
            file_chunks.append(files_to_write)
            command_chunks.append(commands_to_run)

        assert len(file_chunks) == len(command_chunks)
        logger.debug(f"{len(file_chunks)} chunks to execute.")
        all_results = []
        for chunk_idx, (files, commands) in enumerate(
            zip(file_chunks, command_chunks)
        ):
            if len(file_chunks) > 1:
                logger.info(f"Executing chunk {chunk_idx+1}/{len(file_chunks)}")

            _write_maybe_save_error_dir(
                config=config,
                files=files,
                raw_preds=pred_list,
                exec_dir=dir_to_use,
                error_directory=error_directory,
            )
            all_results.extend(execute_commands(commands, config))

            logger.debug("Finished execution, cleaning up...")
            if debug_dir is None:
                cleanup(
                    files,
                    rate_limit=config.write_rate_limit,
                    enable_tqdm=config.display_write_progress,
                )
        all_results = {result[0]: result[1] for result in all_results}
        all_results.update(filtered_results)
        return postprocess_commands(
            raw_preds=pred_list,
            results=all_results,
            postprocessor=postprocessor,
            returned_multiple=preproc_returns_list,
            disable_tqdm=config.disable_tqdm,
            log_freq=config.log_freq,
        )

    if debug_dir is None:
        tmp_dir_loc = os.getenv("EXEC_TMP_DIR")
        if tmp_dir_loc is not None:
            logger.debug("Using %s as temp dir", tmp_dir_loc)
            Path(tmp_dir_loc).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=os.getenv("EXEC_TMP_DIR")
        ) as temp_dir:
            return _run(Path(temp_dir))

    return _run(debug_dir)
