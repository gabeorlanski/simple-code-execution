import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from code_execution.code_trees import safe_ast_parse
from code_execution.eval_dataset import apps
from code_execution.eval_dataset import evaluate_apps
from code_execution.eval_dataset import evaluate_gsm8k

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s - %(levelname)s] %(message)s",
)


def process_ex(ex):
    try:
        solutions = json.loads(ex["solutions"])
    except:
        solutions = None

    try:
        io = json.loads(ex["input_output"])
    except:
        io = None

    return {
        **ex,
        "solutions": solutions,
        "input_output": io,
    }


def main(flags):
    if flags.dataset == "gsm8k":
        dataset = load_dataset("codeparrot/gsm8k", split="test")
        if flags.num_examples:

            print(f"Selecting {flags.num_examples:,} examples")
            dataset = dataset.select(list(range(flags.num_examples)))
        metrics, results = evaluate_gsm8k(
            dataset.to_list(),
            num_workers=4,
            timeout=4,
            execution_kwargs={"log_freq": 10},
        )

    elif flags.dataset == "apps":
        dataset = load_dataset("codeparrot/apps", split="test")
        if flags.num_examples:
            print(f"Selecting {flags.num_examples:,} examples")
            dataset = dataset.select(list(range(flags.num_examples)))
        dataset = dataset.map(
            apps.process_raw_example, load_from_cache_file=False
        )
        dataset = dataset.map(
            lambda x: {
                **x,
                "solutions": [
                    s for s in x["solutions"] if safe_ast_parse(s) is not None
                ],
            }
        )

        dataset = dataset.filter(lambda x: len(x["inputs"]) > 0)
        if flags.sol_per:
            dataset = dataset.map(
                lambda x: {
                    **x,
                    "solutions": x["solutions"][: flags.sol_per],
                }
            )
        print(f"Processing {len(dataset):,} examples")
        metrics, results = evaluate_apps(
            dataset.to_list(),
            num_workers=flags.num_workers,
            timeout=10.0,
            execution_kwargs={"log_freq": 10},
            command_timeout=3.0,
            early_stopping=True,
            max_memory="None",
        )

    out_path = Path("scratch")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "failed.jsonl", "w") as f:
        for r in results:
            for p in r["predictions"]:
                if not p["passed"]:
                    f.write(
                        json.dumps({"problem": r["problem_id"], **p}) + "\n"
                    )
                else:
                    print(r["problem_id"], p["passed"])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["apps", "gsm8k"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--sol_per", type=int, default=None)
    main(parser.parse_args())
