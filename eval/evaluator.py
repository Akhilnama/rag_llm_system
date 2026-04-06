from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_answer(expected: str, actual: str) -> float:
    """
    Compute semantic similarity between expected and actual answer
    """
    emb_expected = model.encode([expected])
    emb_actual = model.encode([actual])

    score = cosine_similarity(emb_expected, emb_actual)[0][0]
    return float(score)


def assign_label(score: float) -> str:
    """
    Assign qualitative label based on score
    """
    if score >= 0.8:
        return "GOOD"
    elif score >= 0.5:
        return "PARTIAL"
    else:
        return "FAIL"


def evaluate_with_label(expected: str, actual: str):
    """
    Full evaluation: score + label
    """
    score = evaluate_answer(expected, actual)
    label = assign_label(score)

    return score, label