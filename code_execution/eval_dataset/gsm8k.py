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


def preprocess(
    pred_dict: Dict, timeout: int = 10, entrypoint: str = "solution"
) -> Executable:
    if "answer" in pred_dict:
        answer = pred_dict["answer"].split("####")[-1]
    elif "target" in pred_dict:
        answer = pred_dict["target"]
    answer = answer.replace(",", "").replace("$", "").replace(" ", "")
    out = []
    for solution in pred_dict["solutions"]:

        out.append(
            Executable(
                files={
                    "main.py": f"{solution}\n\nassert {entrypoint}() == {answer}"
                },
                commands=[Command(["python3", "main.py"], timeout=timeout)],
            )
        )

    return out


def postprocess(pred_dict: Dict, result: List[ExecutionResult]) -> Dict:
    out = []
    for res, pred in zip(result, pred_dict["solutions"]):
        out.append(
            {
                "solution": pred,
                "passed": not res.had_error and not res.timed_out,
                "return_code": res.last_cmd.return_code,
                "stderr": res.last_cmd.stderr,
                "stdout": res.last_cmd.stdout,
            }
        )

    return {
        **{k: pred_dict[k] for k in pred_dict if k != "solutions"},
        "predictions": out,
    }


def evaluate_gsm8k(
    dataset: Dataset,
    num_workers: int,
    timeout: int = 10,
    entrypoint: int = "solution",
    k_vals: List[int] = None,
) -> Tuple[Dict, List[Dict]]:
    logger.debug("Adding index column to dataset")
    dataset = dataset.map(
        lambda ex, idx: {"problem_id": idx, **ex}, with_indices=True
    )

    logger.info("Evaluating GSM8K dataset")

    results = execute_predictions(
        pred_list=dataset,
        config=ExecutionConfig(num_workers=num_workers),
        preprocessor=partial(
            preprocess, timeout=timeout, entrypoint=entrypoint
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
