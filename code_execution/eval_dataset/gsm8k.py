import logging
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple

from code_execution.data_structures import Command
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig
from code_execution.eval_dataset import eval_utils

logger = logging.getLogger(__name__)


def make_executable(
    solution: Dict | List,
    answer: str,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_key: str = "solution",
    default_mem_limit: int = 2 * 1024 * 1024 * 1024,
) -> Executable:
    """Creates an Executable object for evaluating GSM8K solutions.

    Prepares a Python solution for execution by:
    1. Setting memory limits
    2. Extracting the solution code from input
    3. Creating an assertion to validate the answer

    Args:
        solution: Either a dict containing the solution code or a string of code.
                 If dict, the code is extracted using solution_key.
        answer: Expected numeric answer to check against. Will be stripped of
               formatting characters ($, commas, spaces) before comparison.
        timeout: Maximum execution time in seconds before terminating the run.
        entrypoint: Name of the function to call in the solution code.
        solution_key: Key to extract solution from dict if solution is a dict.
        default_mem_limit: Memory limit in bytes (default 2GB).

    Returns:
        Executable object configured to run and validate the solution by calling
        the entrypoint function and comparing its output to the expected answer.

    Raises:
        TypeError: If solution is neither a dict nor a string.
    """
    mem_code = eval_utils.get_mem_limit_code(str(default_mem_limit), "\n\n")
    if isinstance(solution, dict):
        sol_str = solution[solution_key]
    elif isinstance(solution, str):
        sol_str = solution
    else:
        raise TypeError(
            f"Expected solution to be either a dict or a string, got {type(solution)}"
        )
    return Executable(
        files={
            "main.py": f"{mem_code}\n\n{sol_str}\n\nassert {entrypoint}() == {answer}"
        },
        commands=[Command(["python3", "main.py"], timeout=timeout)],
    )


def postprocess_program_result(
    pred: Dict | str, result: ExecutionResult
) -> Dict:
    """Processes execution results into a standardized output format.

    Takes the raw execution result and formats it into a consistent dictionary
    structure containing execution status, error information, and detailed
    timing metrics for each stage of execution.

    Args:
        pred: Original prediction, either as a dict containing metadata or a
             string containing just the solution code.
        result: ExecutionResult from running the code, containing stdout/stderr,
               return codes, timing information, and error flags.

    Returns:
        Dict containing:
        - Original prediction data
        - passed: Boolean indicating if execution succeeded
        - return_code: Exit code from the last command
        - stderr: Error output if any
        - stdout: Standard output from execution
        - timeout: Whether execution timed out
        - had_error: Whether any errors occurred
        - timing: Detailed timing information for each execution phase
    """
    if isinstance(pred, str):
        pred = {"solution": pred}

    t0 = datetime.now()
    did_pass = (
        not result.had_error
        and not result.timed_out
        and result.all_had_return_code(0)
    )
    elapsed = (datetime.now() - t0).total_seconds()
    return {
        **pred,
        "passed": did_pass,
        "return_code": result.last_cmd.return_code,
        "stderr": result.last_cmd.stderr,
        "stdout": result.last_cmd.stdout,
        "timeout": result.timed_out,
        "had_error": result.had_error,
        "timing": {
            "writing": result.writing_time,
            "cleanup": result.cleanup_time,
            "cmd_exec": [c.runtime for c in result.command_results],
            "cmd_eval": [elapsed] * len(result.command_results),
            "preprocess": result.preprocess_time,
            "execution": result.elapsed,
            "postprocess": elapsed,
        },
    }


def preprocess(
    pred_dict: Dict,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_key: str = "solution",
    solution_list_key: str = "solutions",
    default_mem_limit: int = 2 * 1024 * 1024 * 1024,
) -> List[Executable]:
    """Preprocesses GSM8K predictions into executable format.

    Converts a prediction dictionary containing multiple potential solutions
    into a list of Executable objects ready for evaluation. Handles answer
    extraction and normalization, removing formatting characters.

    Args:
        pred_dict: Dictionary containing predictions and expected answer.
                  Must have either 'answer' or 'target' key for the solution,
                  and a list of solutions under solution_list_key.
        timeout: Maximum execution time in seconds per solution attempt.
        entrypoint: Name of the function to call in each solution.
        solution_key: Key for solution code in nested solution dictionaries.
        solution_list_key: Key for accessing the list of solutions to try.
        default_mem_limit: Memory limit in bytes (default 2GB).

    Returns:
        List of Executable objects, one for each solution attempt, each
        configured with appropriate memory limits and timeout settings.

    Raises:
        ValueError: If neither 'answer' nor 'target' key is present in pred_dict.
                   These keys are required to extract the expected answer.
    """
    if "answer" in pred_dict:
        answer = pred_dict["answer"].split("####")[-1]
    elif "target" in pred_dict:
        answer = pred_dict["target"]
    else:
        raise ValueError(
            "Expected 'answer' or 'target' key in the prediction dictionary"
        )
    answer = answer.replace(",", "").replace("$", "").replace(" ", "")
    out = []
    for _i, solution in enumerate(pred_dict[solution_list_key]):
        out.append(
            make_executable(
                solution=solution,
                answer=answer,
                timeout=timeout,
                entrypoint=entrypoint,
                solution_key=solution_key,
                default_mem_limit=default_mem_limit,
            )
        )

    return out


def postprocess(
    pred_dict: Dict,
    result: List[ExecutionResult],
    solution_list_key: str = "solution",
) -> Dict:
    """Processes execution results for multiple solutions.

    Takes a list of execution results from multiple solution attempts and
    processes them into a standardized format, preserving the original
    prediction metadata while adding detailed execution results.

    Args:
        pred_dict: Original prediction dictionary containing problem metadata
                  and multiple solution attempts.
        result: List of ExecutionResults from running each solution attempt,
               containing execution status and timing information.
        solution_list_key: Key for accessing the list of solutions in pred_dict.

    Returns:
        Dict containing:
        - All original prediction metadata (except solutions list)
        - predictions: List of processed results for each solution attempt,
          each containing execution status, outputs, and timing information
    """
    out = []
    for res, pred in zip(result, pred_dict[solution_list_key]):
        out.append(postprocess_program_result(pred=pred, result=res))

    return {
        **{k: pred_dict[k] for k in pred_dict if k != solution_list_key},
        "predictions": out,
    }


def evaluate(
    predictions: List[Dict],
    num_workers: int,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    k_vals: List[int] = None,
    execution_kwargs: Dict = None,
    default_mem_limit: int = 2 * 1024 * 1024 * 1024,
) -> Tuple[Dict, List[Dict]]:
    """Evaluates a batch of GSM8K predictions.

    Coordinates the parallel execution and evaluation of multiple GSM8K problems,
    each with potentially multiple solution attempts. Calculates pass@k metrics
    and aggregates timing information.

    Args:
        predictions: List of prediction dictionaries to evaluate. Each dict
                    should contain problem metadata and multiple solution attempts.
        num_workers: Number of parallel workers for execution. Higher values
                    increase throughput but also memory usage.
        timeout: Maximum execution time in seconds per solution attempt.
        entrypoint: Name of the function to call in each solution.
        solution_str_key: Key for solution code in nested solution dictionaries.
        solution_list_key: Key for accessing the list of solutions to try.
        k_vals: List of k values for calculating pass@k metrics. Default is
               [1, 5, 10]. Values larger than the number of solutions are skipped.
        execution_kwargs: Additional kwargs passed to ExecutionConfig for
                        customizing execution behavior.
        default_mem_limit: Memory limit in bytes (default 2GB).

    Returns:
        Tuple containing:
        - metrics: Dict with pass@k metrics and aggregated timing information
        - results: List of detailed execution results for each problem
    """
    logger.debug("Adding index column to dataset")
    for idx in range(len(predictions)):
        predictions[idx]["problem_id"] = idx

    logger.info("Evaluating GSM8K dataset")

    exec_results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(
            num_workers=num_workers, **(execution_kwargs or {})
        ),
        preprocessor=partial(
            preprocess,
            timeout=timeout,
            entrypoint=entrypoint,
            solution_key=solution_str_key,
            solution_list_key=solution_list_key,
            default_mem_limit=default_mem_limit,
        ),
        postprocessor=partial(postprocess, solution_list_key=solution_list_key),
        preproc_returns_list=True,
    )

    num_samples = 1
    pass_counts = []
    for r in exec_results.results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {
        "percent_passed": sum(pc > 0 for pc in pass_counts) / len(pass_counts),
        **exec_results.timing_dict,
    }
    for k in k_vals or [1, 5, 10]:
        if k > num_samples:
            break
        metrics[f"pass@{k}"] = float(
            estimate_pass_at_k(
                [num_samples] * len(pass_counts), pass_counts, k
            ).mean()
        )

    return metrics, exec_results.results
