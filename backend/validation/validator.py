"""
Validation and filtering layer.

Applies two checks to each candidate question:
  1. Relevance  — semantic similarity between question and source chunk (local, free)
  2. Answerability — LLM verifies the correct answer is grounded in the source (optional)
"""

from backend.generation.question import Question
from backend.validation.scorer import question_chunk_similarity
from backend.validation.answerability import is_answerable

_PASS = "\033[92m✓ PASS\033[0m"
_FAIL = "\033[91m✗ FAIL\033[0m"
_SKIP = "\033[90m– SKIP\033[0m"


def validate(
    question: Question,
    similarity_threshold: float = 0.25,
    check_answerability: bool = True,
) -> tuple[bool, dict]:
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
    _print_report(question, report, passed, check_answerability)
    return passed, report


def _print_report(question: Question, report: dict, passed: bool, check_answerability: bool):
    status = _PASS if passed else _FAIL
    sim_status = _PASS if report["relevance_passed"] else _FAIL

    if check_answerability:
        ans_status = _PASS if report["answerability_passed"] else _FAIL
    else:
        ans_status = _SKIP

    q_preview = question.question[:80] + "..." if len(question.question) > 80 else question.question

    print(
        f"\n  {status} {q_preview}\n"
        f"         similarity : {report['similarity']:.4f}  {sim_status}\n"
        f"       answerability: {ans_status}"
    )


def filter_candidates(
    candidates: list[Question],
    similarity_threshold: float = 0.25,
    check_answerability: bool = True,
) -> list[tuple[Question, dict]]:
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
    total_candidates = sum(len(c) for c in all_candidates)
    print(f"\n{'─' * 60}")
    print(f"  VALIDATION  |  {total_candidates} candidates  |  threshold={similarity_threshold}  |  answerability={'on' if check_answerability else 'off'}")
    print(f"{'─' * 60}")

    validated = []
    for chunk_i, candidates in enumerate(all_candidates, 1):
        if max_questions and len(validated) >= max_questions:
            break
        print(f"\n  Chunk {chunk_i} ({len(candidates)} candidates)")
        passing = filter_candidates(candidates, similarity_threshold, check_answerability)
        validated.extend(q for q, _ in passing)

    print(f"\n{'─' * 60}")
    print(f"  {len(validated)}/{total_candidates} questions passed validation")
    print(f"{'─' * 60}\n")

    return validated[:max_questions] if max_questions else validated
