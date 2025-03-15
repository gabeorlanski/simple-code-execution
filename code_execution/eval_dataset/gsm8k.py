import logging
from functools import partial
from typing import Dict, List, Tuple

from code_execution.data_structures import Command
from code_execution.data_structures import Executable
from code_execution.data_structures import ExecutionResult
from code_execution.entrypoints import execute_predictions
from code_execution.eval_dataset.metrics import estimate_pass_at_k
from code_execution.execution import ExecutionConfig

logger = logging.getLogger(__name__)


def make_executable(
    solution: Dict | List,
    answer: str,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_key: str = "solution",
) -> Executable:
    if isinstance(solution, dict):
        sol_str = solution[solution_key]
    elif isinstance(solution, str):
        sol_str = solution
    else:
        raise TypeError(
            f"Expected solution to be either a dict or a string, got {type(solution)}"
        )
    return Executable(
        files={"main.py": f"{sol_str}\n\nassert {entrypoint}() == {answer}"},
        commands=[Command(["python3", "main.py"], timeout=timeout)],
    )


def postprocess_program_result(
    pred: Dict | str, result: ExecutionResult
) -> bool:
    if isinstance(pred, str):
        pred = {"solution": pred}
    return {
        **pred,
        "passed": not result.had_error and not result.timed_out,
        "return_code": result.last_cmd.return_code,
        "stderr": result.last_cmd.stderr,
        "stdout": result.last_cmd.stdout,
        "elapsed": result.elapsed,
        "writing_time": result.writing_time,
        "cleanup_time": result.cleanup_time,
        "preprocess_time": result.preprocess_time,
    }


def preprocess(
    pred_dict: Dict,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_key: str = "solution",
    solution_list_key: str = "solutions",
) -> Executable:
    """Preprocesses the GSM8K prediction into an executable. Expected keys in the pred_dict are:
    - "answer" or "target": the expected output
    - "solutions": a list of strings or dicts containing the solution code.
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
            )
        )

    return out


def postprocess(
    pred_dict: Dict,
    result: List[ExecutionResult],
    solution_list_key: str = "solution",
) -> Dict:
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
    entrypoint: int = "solution",
    solution_str_key: str = "solution",
    solution_list_key: str = "solutions",
    k_vals: List[int] = None,
) -> Tuple[Dict, List[Dict]]:
    logger.debug("Adding index column to dataset")
    for idx in range(len(predictions)):
        predictions[idx]["problem_id"] = idx

    logger.info("Evaluating GSM8K dataset")

    exec_results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(num_workers=num_workers),
        preprocessor=partial(
            preprocess,
            timeout=timeout,
            entrypoint=entrypoint,
            solution_key=solution_str_key,
            solution_list_key=solution_list_key,
        ),
        postprocessor=partial(postprocess, solution_list_key=solution_list_key),
        preproc_returns_list=True,
    )

    num_samples = 1
    pass_counts = []
    for r in exec_results.results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {**exec_results.timing_dict}
    for k in k_vals or [1, 5, 10]:
        if k > num_samples:
            break
        metrics[f"pass@{k}"] = float(
            estimate_pass_at_k(
                [num_samples] * len(pass_counts), pass_counts, k
            ).mean()
        )

    return metrics, exec_results.results
