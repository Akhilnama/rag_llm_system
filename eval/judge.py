from typing import Dict


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return f"""
You are an expert evaluator for LLM systems.

Evaluate the following answer based ONLY on the given context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Evaluate on:

1. Faithfulness (Is answer grounded in context?)
2. Relevance (Does it answer the question?)
3. Correctness (Is it factually correct?)

Return STRICT JSON:

{{
    "faithfulness": score between 0 and 1,
    "relevance": score between 0 and 1,
    "correctness": score between 0 and 1,
    "final_score": average,
    "label": "GOOD" or "PARTIAL" or "FAIL"
}}
"""

import json


def evaluate_with_llm(llm, question: str, context: str, answer: str) -> Dict:
    prompt = build_judge_prompt(question, context, answer)

    response = llm.generate(prompt)

    try:
        result = json.loads(response)
    except:
        result = {
            "error": "Failed to parse",
            "raw_output": response
        }

    return result

def detect_hallucination(llm_eval: Dict) -> bool:
    if "faithfulness" in llm_eval:
        return llm_eval["faithfulness"] < 0.5
    return False