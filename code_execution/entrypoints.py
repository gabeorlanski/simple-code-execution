"""Module for entrypoints for code execution."""

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from code_execution.configs import ExecutionConfig
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.data_structures import OverallExecutionResults
from code_execution.execution import execute_commands
from code_execution.file_writing import cleanup
from code_execution.file_writing import write_executables
from code_execution.processing import PredTimingsCollection
from code_execution.processing import default_postprocessor
from code_execution.processing import postprocess_commands
from code_execution.processing import preprocess_commands

logger = logging.getLogger(__name__)


@dataclass
class ChunkExecutionResult:
    """Stores the execution result for a chunk of commands."""

    results: Dict[Tuple[int, int], ExecutionResult]
    write_elapsed: float
    exec_elapsed: float
    pure_exec_elapsed: float
    write_timings: Dict[str, float]
    cleanup_timings: Dict[str, float]


def _write_maybe_save_error_dir(
    config, files, raw_preds, exec_dir: Path, error_directory: Optional[Path]
):
    try:
        times = write_executables(
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
    return times


def _preproc_step(
    config: ExecutionConfig,
    pred_list: List[Dict],
    preprocessor: Callable[[Dict], Union[Executable, List[Executable]]],
    execution_dir: Path,
    preproc_returns_list: bool = False,
    preproc_batch_size: int = 1,
) -> Tuple:
    """Preprocess commands and measure timing.

    Args:
        config: The config for execution.
        pred_list: The list of predictions to execute.
        preprocessor: The preprocessor function to create files and commands.
        execution_dir: Directory where execution will take place.
        preproc_returns_list: Whether preprocessor returns one or many items.
        preproc_batch_size: The batch size for preprocessing.

    Returns:
        Tuple: (files_to_write, commands_to_run, filtered_results, preproc_timings)
    """
    logger.debug("Preprocessing commands")
    preproc_start = datetime.now()

    *preproc_res, preproc_timings = preprocess_commands(
        config=config,
        dir_to_use=execution_dir,
        pred_list=pred_list,
        preprocessor=preprocessor,
        preproc_returns_list=preproc_returns_list,
        batch_size=preproc_batch_size,
    )

    preproc_elapsed = (datetime.now() - preproc_start).total_seconds()
    timings = (preproc_timings, preproc_elapsed)

    return preproc_res, timings


def _create_execution_chunks(
    config: ExecutionConfig,
    files_to_write: List,
    commands_to_run: List,
) -> tuple:
    """Split files and commands into manageable chunks.

    Args:
        config: The config for execution.
        files_to_write: List of files to write.
        commands_to_run: List of commands to run.

    Returns:
        tuple: (file_chunks, command_chunks)
    """
    file_chunks = []
    command_chunks = []

    if (
        config.max_execute_at_once > 0
        and len(commands_to_run) > config.max_execute_at_once
    ):
        logger.info(
            f"Executing {len(commands_to_run):,} commands "
            f"in chunks of {config.max_execute_at_once:,}"
        )

        if config.max_execute_at_once == 1:
            logger.warning(
                "max_execute_at_once is set to 1, this will be slow."
            )
            file_chunks = [[f] for f in files_to_write]
            command_chunks = [[c] for c in commands_to_run]
        else:
            for i in range(0, len(commands_to_run), config.max_execute_at_once):
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

    return file_chunks, command_chunks


def _process_single_chunk(
    chunk_idx: int,
    total_chunks: int,
    files: List,
    commands: List,
    config: ExecutionConfig,
    pred_list: List[Dict],
    execution_dir: Path,
    debug_dir: Optional[Path] = None,
    error_directory: Optional[Path] = None,
) -> ChunkExecutionResult:
    """Process a single chunk of files and commands.

    Args:
        chunk_idx: Index of the current chunk.
        total_chunks: Total number of chunks.
        files: List of files in this chunk.
        commands: List of commands in this chunk.
        config: The config for execution.
        pred_list: The list of predictions.
        execution_dir: Directory where execution will take place.
        debug_dir: Directory to save all files for debugging.
        error_directory: Directory to save errors to.

    Returns:
        Tuple: (results, (write_elapsed, exec_elapsed, pure_exec_elapsed, write_timings, cleanup_timings))
    """
    if total_chunks > 1:
        logger.info(f"Executing chunk {chunk_idx+1}/{total_chunks}")

    # Write files
    write_start = datetime.now()
    write_timings = _write_maybe_save_error_dir(
        config=config,
        files=files,
        raw_preds=pred_list,
        exec_dir=execution_dir,
        error_directory=error_directory,
    )
    write_elapsed = (datetime.now() - write_start).total_seconds()

    # Execute commands
    exec_elapsed, pure_exec_elapsed, results = execute_commands(
        commands, config
    )

    # Clean up if in debug mode
    logger.debug("Finished execution, cleaning up...")
    cleanup_timings = {}
    if debug_dir is not None:
        cleanup_timings = cleanup(
            files,
            rate_limit=config.write_rate_limit,
            enable_tqdm=config.display_write_progress,
        )

    return ChunkExecutionResult(
        results=results,
        write_elapsed=write_elapsed,
        exec_elapsed=exec_elapsed,
        pure_exec_elapsed=pure_exec_elapsed,
        write_timings=write_timings,
        cleanup_timings=cleanup_timings,
    )


def _run_execution_workflow(
    config: ExecutionConfig,
    pred_list: List[Dict],
    preprocessor: Callable[[Dict], Union[Executable, List[Executable]]],
    postprocessor: Callable[[Dict, Dict], Dict],
    execution_dir: Path,
    debug_dir: Optional[Path] = None,
    preproc_returns_list: bool = False,
    preproc_batch_size: int = 1,
    error_directory: Optional[Path] = None,
) -> OverallExecutionResults:
    """Run the execution workflow in a given directory.

    Args:
        config: The config for execution.
        pred_list: The list of predictions to execute.
        preprocessor: The preprocessor function to create files and commands.
        postprocessor: The postprocessor function for processing results.
        execution_dir: Directory where execution will take place.
        debug_dir: Directory to save all files for debugging.
        preproc_returns_list: Whether preprocessor returns one or many items.
        preproc_batch_size: The batch size for preprocessing.
        error_directory: Directory to save errors to.

    Returns:
        OverallExecutionResults: The results of the execution.
    """
    logger.debug(f"Using {execution_dir} as execution directory")
    start_time = datetime.now()

    # Step 1: Preprocess commands
    preproc_res, (preproc_timings, preproc_elapsed) = _preproc_step(
        config=config,
        pred_list=pred_list,
        preprocessor=preprocessor,
        execution_dir=execution_dir,
        preproc_returns_list=preproc_returns_list,
        preproc_batch_size=preproc_batch_size,
    )
    files_to_write, commands_to_run, filtered_results = preproc_res
    # Step 2: Create execution chunks
    file_chunks, command_chunks = _create_execution_chunks(
        config=config,
        files_to_write=files_to_write,
        commands_to_run=commands_to_run,
    )

    # Step 3: Process each chunk
    chunk_results = []
    for chunk_idx, (files, commands) in enumerate(
        zip(file_chunks, command_chunks)
    ):
        chunk_results.append(
            _process_single_chunk(
                chunk_idx=chunk_idx,
                total_chunks=len(file_chunks),
                files=files,
                commands=commands,
                config=config,
                pred_list=pred_list,
                execution_dir=execution_dir,
                debug_dir=debug_dir,
                error_directory=error_directory,
            )
        )

    # Step 4: Combine results and postprocess
    results_dict = {}
    write_elapsed = exec_elapsed = pure_exec_elapsed = 0
    write_timings = {}
    cleanup_timings = {}
    for result in chunk_results:

        results_dict.update({r[0]: r[1] for r in result.results})
        write_elapsed += result.write_elapsed
        exec_elapsed += result.exec_elapsed
        pure_exec_elapsed += result.pure_exec_elapsed
        write_timings.update(result.write_timings)
        cleanup_timings.update(result.cleanup_timings)
    results_dict.update(filtered_results)
    timings = PredTimingsCollection(
        preprocess_time=preproc_timings,
        writing_time=write_timings,
        cleanup_time=cleanup_timings,
    )

    # Step 5: Postprocess the results
    post_start = datetime.now()
    postprocessed = postprocess_commands(
        raw_preds=pred_list,
        results=results_dict,
        postprocessor=postprocessor,
        returned_multiple=preproc_returns_list,
        timings=timings,
        disable_tqdm=config.disable_tqdm,
        log_freq=config.log_freq,
    )
    post_elapsed = (datetime.now() - post_start).total_seconds()
    net_elapsed = (datetime.now() - start_time).total_seconds()

    return OverallExecutionResults(
        results=postprocessed,
        net_time=net_elapsed,
        pure_exec_time=pure_exec_elapsed,
        execution_time=exec_elapsed,
        writing_time=write_elapsed,
        postprocessing_time=post_elapsed,
        preprocessing_time=preproc_elapsed,
    )


def execute_predictions(
    config: ExecutionConfig,
    pred_list: List[Dict],
    preprocessor: Callable[[Dict], Union[Executable, List[Executable]]],
    postprocessor: Callable[[Dict, Dict], Dict] = None,
    debug_dir: Optional[Path] = None,
    preproc_returns_list: bool = False,
    preproc_batch_size: int = 1,
    error_directory: Optional[Path] = None,
) -> OverallExecutionResults:
    """Executes the program predictions.

    First preprocesses the commands to run, writes them to disk, then executes
    them, and finally postprocesses the results.

    Args:
        config: The config for execution.
        pred_list: The list of predictions to execute.
        preprocessor: The preprocessor function to create files and commands.
        postprocessor: The postprocessor function for processing results.
        debug_dir: Directory to save all files for debugging.
        preproc_returns_list: Whether preprocessor returns one or many items.
        preproc_batch_size: The batch size for preprocessing.
        error_directory: Directory to save errors to.

    Returns:
        OverallExecutionResults: The results of the execution.
    """
    # Use default postprocessor if none provided
    if postprocessor is None:
        logger.info("Using default postprocessor")
        postprocessor = default_postprocessor

    logger.debug(f"Starting execution with {len(pred_list):,} predictions")

    # Either use debug directory or create a temporary one
    if debug_dir is None:
        # Check for environment variable specifying temp directory
        tmp_dir_loc = os.getenv("EXEC_TMP_DIR")
        if tmp_dir_loc is not None:
            logger.debug(f"Using {tmp_dir_loc} as temp dir")
            Path(tmp_dir_loc).mkdir(parents=True, exist_ok=True)

        # Create temporary directory and run workflow
        with tempfile.TemporaryDirectory(dir=tmp_dir_loc) as temp_dir:
            return _run_execution_workflow(
                config=config,
                pred_list=pred_list,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                execution_dir=Path(temp_dir),
                debug_dir=None,
                preproc_returns_list=preproc_returns_list,
                preproc_batch_size=preproc_batch_size,
                error_directory=error_directory,
            )

    # Use provided debug directory
    return _run_execution_workflow(
        config=config,
        pred_list=pred_list,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        execution_dir=debug_dir,
        debug_dir=debug_dir,
        preproc_returns_list=preproc_returns_list,
        preproc_batch_size=preproc_batch_size,
        error_directory=error_directory,
    )
