from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    return float(cosine_similarity(emb1, emb2)[0][0])


def evaluate_retrieval_semantic(
    context_chunks: List[Dict],
    expected_answer: str
) -> Dict:
    """
    Evaluate retrieval using full context (semantic match)
    """
    context_text = " ".join([c["text"] for c in context_chunks])

    score = compute_similarity(context_text, expected_answer)

    return {
        "retrieval_score": score,
        "label": assign_label(score)
    }


def evaluate_retrieval_weighted(
    context_chunks: List[Dict],
    expected_answer: str
) -> Dict:
    """
    Weighted retrieval evaluation:
    - Top chunks get higher importance
    - Better reflects real RAG behavior
    """
    if not context_chunks:
        return {"retrieval_score": 0.0, "label": "FAIL"}

    scores = []
    weights = []

    for i, chunk in enumerate(context_chunks):
        sim = compute_similarity(chunk["text"], expected_answer)

        weight = 1 / (i + 1)  # rank-based weighting
        scores.append(sim * weight)
        weights.append(weight)

    final_score = sum(scores) / sum(weights)

    return {
        "retrieval_score": float(final_score),
        "label": assign_label(final_score)
    }


def evaluate_top_chunk(
    context_chunks: List[Dict],
    expected_answer: str
) -> Dict:
    """
    Evaluate only the top retrieved chunk
    (Very important diagnostic signal)
    """
    if not context_chunks:
        return {"top_chunk_score": 0.0, "label": "FAIL"}

    top_chunk = context_chunks[0]["text"]
    score = compute_similarity(top_chunk, expected_answer)

    return {
        "top_chunk_score": score,
        "label": assign_label(score)
    }


def assign_label(score: float) -> str:
    if score >= 0.7:
        return "GOOD"
    elif score >= 0.4:
        return "PARTIAL"
    else:
        return "FAIL"