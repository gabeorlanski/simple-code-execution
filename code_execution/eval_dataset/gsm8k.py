import logging
from functools import partial
from typing import Dict, List, Tuple

from datasets import Dataset

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
    elif isinstance(solution, list):
        sol_str = solution
    else:
        raise TypeError(
            f"Expected solution to be either a dict or a string, got {type(solution)}"
        )
    return Executable(
        files={"main.py": f"{sol_str}\n\nassert {entrypoint}() == {answer}"},
        commands=[Command(["python3", "main.py"], timeout=timeout)],
    )


def postprocess_program_result(pred: str, result: ExecutionResult) -> bool:
    return {
        "solution": pred,
        "passed": not result.had_error and not result.timed_out,
        "return_code": result.last_cmd.return_code,
        "stderr": result.last_cmd.stderr,
        "stdout": result.last_cmd.stdout,
    }


def preprocess(
    pred_dict: Dict,
    timeout: int = 10,
    entrypoint: str = "solution",
    solution_key: str = "solution",
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
    for _i, solution in enumerate(pred_dict["solutions"]):
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


def postprocess(pred_dict: Dict, result: List[ExecutionResult]) -> Dict:
    out = []
    for res, pred in zip(result, pred_dict["solutions"]):
        out.append(postprocess_program_result(pred=pred, result=res))

    return {
        **{k: pred_dict[k] for k in pred_dict if k != "solutions"},
        "predictions": out,
    }


def evaluate(
    predictions: List[Dict],
    num_workers: int,
    timeout: int = 10,
    entrypoint: int = "solution",
    solution_key: str = "solution",
    k_vals: List[int] = None,
) -> Tuple[Dict, List[Dict]]:
    logger.debug("Adding index column to dataset")
    for idx in range(len(predictions)):
        predictions[idx]["problem_id"] = idx

    logger.info("Evaluating GSM8K dataset")

    results = execute_predictions(
        pred_list=predictions,
        config=ExecutionConfig(num_workers=num_workers),
        preprocessor=partial(
            preprocess,
            timeout=timeout,
            entrypoint=entrypoint,
            solution_key=solution_key,
        ),
        postprocessor=postprocess,
        preproc_returns_list=True,
    )

    num_samples = 1
    pass_counts = []
    for r in results:
        pass_counts.append(sum([p["passed"] for p in r["predictions"]]))
        num_samples = max(num_samples, len(r["predictions"]))

    metrics = {}
    for k in k_vals or [1, 5, 10]:
        if k > num_samples:
            break
        metrics[f"pass@{k}"] = float(
            estimate_pass_at_k(
                [num_samples] * len(pass_counts), pass_counts, k
            ).mean()
        )

    return metrics, results
