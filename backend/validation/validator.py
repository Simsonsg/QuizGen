"""
Validation and filtering layer.

Applies up to three checks on each candidate question:
  1. Relevance      — semantic similarity between question and source chunk (always on)
  2. Answerability  — LLM verifies correct answer is grounded in source (optional)
  3. Cognitive level — LLM independently classifies Bloom's level and checks it
                       matches the intended level (optional)
"""

from backend.generation.question import Question
from backend.validation.scorer import question_chunk_similarity
from backend.validation.answerability import is_answerable
from backend.validation.cognitive_check import check_cognitive_level

_PASS = "\033[92m✓ PASS\033[0m"
_FAIL = "\033[91m✗ FAIL\033[0m"
_SKIP = "\033[90m– SKIP\033[0m"


def validate(
    question: Question,
    similarity_threshold: float = 0.25,
    check_answerability: bool = False,
    check_cognitive: bool = False,
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

    # --- Cognitive-level check ---
    if check_cognitive:
        matches, actual_level = check_cognitive_level(
            question.question,
            question.options,
            question.cognitive_level,
        )
        report["cognitive_passed"] = matches
        report["cognitive_actual"] = actual_level
    else:
        report["cognitive_passed"] = True
        report["cognitive_actual"] = None

    passed = report["relevance_passed"] and report["answerability_passed"] and report["cognitive_passed"]
    question.similarity_score = report["similarity"]
    _print_report(question, report, passed, check_answerability, check_cognitive)
    return passed, report


def _print_report(question: Question, report: dict, passed: bool, check_answerability: bool, check_cognitive: bool):
    status = _PASS if passed else _FAIL
    sim_status = _PASS if report["relevance_passed"] else _FAIL

    ans_status = (_PASS if report["answerability_passed"] else _FAIL) if check_answerability else _SKIP

    if check_cognitive:
        actual = report.get("cognitive_actual") or "unknown"
        cog_status = (_PASS if report["cognitive_passed"] else _FAIL)
        cog_line = f"  cognitive level: {question.cognitive_level} → {actual}  {cog_status}"
    else:
        cog_line = f"  cognitive level: {_SKIP}"

    q_preview = question.question[:80] + "..." if len(question.question) > 80 else question.question

    print(
        f"\n  {status} {q_preview}\n"
        f"         similarity : {report['similarity']:.4f}  {sim_status}\n"
        f"       answerability: {ans_status}\n"
        f"     {cog_line}"
    )


def filter_candidates(
    candidates: list[Question],
    similarity_threshold: float = 0.25,
    check_answerability: bool = False,
    check_cognitive: bool = False,
) -> list[tuple[Question, dict]]:
    passed = []
    for q in candidates:
        ok, report = validate(q, similarity_threshold, check_answerability, check_cognitive)
        if ok:
            passed.append((q, report))
    return passed


def filter_all(
    all_candidates: list[list[Question]],
    similarity_threshold: float = 0.25,
    check_answerability: bool = False,
    check_cognitive: bool = False,
    max_questions: int = 20,
) -> list[Question]:
    total_candidates = sum(len(c) for c in all_candidates)
    checks = " | ".join(filter(None, [
        f"similarity≥{similarity_threshold}",
        "answerability=on" if check_answerability else None,
        "cognitive=on" if check_cognitive else None,
    ]))
    print(f"\n{'─' * 60}")
    print(f"  VALIDATION  |  {total_candidates} candidates  |  {checks}")
    print(f"{'─' * 60}")

    validated = []
    for chunk_i, candidates in enumerate(all_candidates, 1):
        if max_questions and len(validated) >= max_questions:
            break
        print(f"\n  Chunk {chunk_i} ({len(candidates)} candidates)")
        passing = filter_candidates(candidates, similarity_threshold, check_answerability, check_cognitive)
        validated.extend(q for q, _ in passing)

    print(f"\n{'─' * 60}")
    print(f"  {len(validated)}/{total_candidates} questions passed validation")
    print(f"{'─' * 60}\n")

    return validated[:max_questions] if max_questions else validated
