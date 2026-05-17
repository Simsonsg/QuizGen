"""
Baseline single-pass question generator.

Used as the comparison condition for the dissertation evaluation.

Unlike the controlled pipeline, this generator:
  - Uses a simple, unconditioned prompt (no difficulty or cognitive-level guidance)
  - Generates exactly one question per chunk (no multi-candidate pool)
  - Applies no validation or filtering

This mirrors how a naive LLM prompting approach would work, allowing direct
comparison against the controlled multi-stage pipeline.
"""

import json
import re
from backend.llm import complete
from backend.generation.question import Question

_SYSTEM_PROMPT = """You are an educational assistant.
Generate a multiple-choice question based on the provided text.
Always respond with valid JSON only. No prose, no markdown, no code fences."""

_USER_TEMPLATE = """Generate one multiple-choice question from the following text.

The question must have exactly 4 options (A, B, C, D) with one correct answer.

Source text:
\"\"\"
{chunk}
\"\"\"

Respond with a single JSON object with these exact keys:
  "question": string
  "options": {{"A": string, "B": string, "C": string, "D": string}}
  "answer": "A" | "B" | "C" | "D"

JSON object:"""


def generate_baseline(chunk: str) -> list[Question]:
    """
    Generate a single unconditioned question from a text chunk.

    Returns a list with 0 or 1 Question objects.
    Difficulty and cognitive_level are set to "unspecified" to distinguish
    baseline questions from pipeline questions in logged session data.
    """
    prompt = _USER_TEMPLATE.format(chunk=chunk)
    raw = complete(_SYSTEM_PROMPT, prompt, max_tokens=512)
    return _parse_response(raw, chunk)


def generate_all_baseline(chunks: list[str], max_questions: int = 10) -> list[Question]:
    """
    Generate one baseline question per chunk sequentially, up to max_questions.
    Stops as soon as max_questions is reached.
    """
    questions = []
    for chunk in chunks:
        if len(questions) >= max_questions:
            break
        questions.extend(generate_baseline(chunk))
    return questions[:max_questions]


def _parse_response(raw: str, chunk: str) -> list[Question]:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # Accept either a JSON object or a single-element array
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    if not parsed:
        return []

    try:
        q = Question(
            question=parsed["question"],
            options=parsed["options"],
            answer=parsed["answer"].upper(),
            difficulty="unspecified",
            cognitive_level="unspecified",
            source_chunk=chunk,
        )
        return [q]
    except (KeyError, AttributeError):
        return []
