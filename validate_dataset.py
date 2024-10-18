import json

from datasets import load_dataset

from code_execution.datasets.gsm8k import evaluate_gsm8k


def main():
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    def proc(ex):
        if "target" in ex:
            answer = ex["target"]
        else:
            answer = ex["answer"].split("#")[-1]
        answer = answer.replace(",", "").replace("$", "").replace(" ", "")
        answer = f"def solution():\n    return {answer}"
        return {"solutions": [answer]}

    dataset = dataset.map(proc)
    metrics, results = evaluate_gsm8k(dataset, num_workers=4)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
