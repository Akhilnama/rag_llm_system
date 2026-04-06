import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "eval_dataset.json"
import json
from src.pipeline import run_query
from eval.evaluator import evaluate_with_label

def run_evaluation():
    with open(f"{DATA_PATH}", "r") as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]

        output = run_query(question)
        actual = output["answer"]

        score, label = evaluate_with_label(expected, actual)

        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "score": score
        })

        print(f"Q: {question}")
        print(f"Score: {score:.2f} | Label: {label}")
        print("-" * 40)

    return results


if __name__ == "__main__":
    run_evaluation()