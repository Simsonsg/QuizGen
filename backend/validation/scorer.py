"""
Semantic similarity scoring using sentence embeddings
"""

from sentence_transformers import SentenceTransformer, util

_MODEL = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.
    Returns a float in [0, 1].
    """
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score)


def question_chunk_similarity(question: str, chunk: str) -> float:
    """
    Measure how semantically relevant a question is to its source chunk.
    """
    return similarity(question, chunk)
