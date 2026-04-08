import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "eval_dataset.json"
import json
from src.pipeline import run_query
from eval.evaluator import evaluate_with_label

from eval.judge import evaluate_with_llm, detect_hallucination
from src.pipeline import llm

from eval.retrieval_eval import (
    evaluate_retrieval_weighted,
    evaluate_top_chunk
)

def run_evaluation():
    with open(f"{DATA_PATH}", "r") as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]

        output = run_query(question)
        actual = output["answer"]
        # LLM as judge
        context_text = "\n".join([c["text"] for c in output["context"]])

        llm_eval = evaluate_with_llm(
            llm,
            question,
            context_text,
            actual
        )
        print("LLM Eval", llm_eval)

        retrieval_eval = evaluate_retrieval_weighted(
            output["context"],
            item["expected_answer"]
        )
        print("Retrieval (weighted):", retrieval_eval)

        
        top_chunk_eval = evaluate_top_chunk(
            output["context"],
            item["expected_answer"]
        )

        print("Top Chunk:", top_chunk_eval)
        
        

        is_hallucination = detect_hallucination(llm_eval)

        print("Hallucination", is_hallucination)
        

        

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