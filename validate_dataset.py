import json

from datasets import load_dataset

from code_execution.eval_dataset.apps import evaluate_apps


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


def main():
    dataset = load_dataset("codeparrot/apps", split="test")

    metrics, results = evaluate_apps(
        dataset,
        num_workers=4,
        timeout=4,
        execution_kwargs={"log_freq": 1},
        command_timeout=3.0,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
