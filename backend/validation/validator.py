"""
Validation and filtering layer.

Applies two checks to each candidate question:
  1. Relevance  — semantic similarity between question and source chunk (local, free)
  2. Answerability — LLM verifies the correct answer is grounded in the source (optional)

Difficulty consistency is not re-judged here; it is enforced at generation time
via prompt conditioning and can be evaluated separately during the study.
"""

from backend.generation.question import Question
from backend.validation.scorer import question_chunk_similarity
from backend.validation.answerability import is_answerable


def validate(
    question: Question,
    similarity_threshold: float = 0.25,
    check_answerability: bool = True,
) -> tuple[bool, dict]:
    """
    Run all validation checks on a single Question.

    Args:
        question: The candidate question to validate.
        similarity_threshold: Minimum cosine similarity score to pass relevance check.
        check_answerability: Whether to run the LLM answerability check.

    Returns:
        (passed: bool, report: dict) where report contains per-check scores/results.
    """
    report = {}

    # --- Relevance check ---
    sim_score = question_chunk_similarity(question.question, question.source_chunk)
    report["similarity"] = round(sim_score, 4)
    report["relevance_passed"] = sim_score >= similarity_threshold

    # --- Answerability check ---
    if check_answerability:
        answerable = is_answerable(
            question.question,
            question.correct_answer_text(),
            question.source_chunk,
        )
        report["answerability_passed"] = answerable
    else:
        report["answerability_passed"] = True

    passed = report["relevance_passed"] and report["answerability_passed"]
    return passed, report


def filter_candidates(
    candidates: list[Question],
    similarity_threshold: float = 0.25,
    check_answerability: bool = True,
) -> list[tuple[Question, dict]]:
    """
    Validate a list of candidate questions and return only those that pass.

    Returns:
        List of (question, report) tuples for questions that passed all checks.
    """
    passed = []
    for q in candidates:
        ok, report = validate(q, similarity_threshold, check_answerability)
        if ok:
            passed.append((q, report))
    return passed


def filter_all(
    all_candidates: list[list[Question]],
    similarity_threshold: float = 0.25,
    check_answerability: bool = True,
    max_questions: int = 20,
) -> list[Question]:
    """
    Validate candidates across all chunks and return a flat list of passing questions.
    Stops early once max_questions have been collected.
    """
    validated = []
    for candidates in all_candidates:
        if max_questions and len(validated) >= max_questions:
            break
        passing = filter_candidates(candidates, similarity_threshold, check_answerability)
        validated.extend(q for q, _ in passing)
    return validated[:max_questions] if max_questions else validated


