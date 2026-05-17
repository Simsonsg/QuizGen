"""
Controlled multi-candidate question generation.

For each chunk, the LLM is prompted to generate N candidate questions with:
  - explicit difficulty conditioning (easy / medium / hard)
  - Bloom's taxonomy cognitive-level conditioning
  - source restriction (questions must be answerable from the provided text only)

The LLM is asked to return structured JSON, which is parsed into Question objects.
"""

import json
import re
from backend.llm import complete
from backend.generation.question import Question

_SYSTEM_PROMPT = """You are an expert educational assessment designer.
Your task is to generate multiple-choice questions strictly based on the provided source text.
Do not use any external knowledge — every question must be directly answerable from the text.
Always respond with valid JSON only. No prose, no markdown, no code fences."""

_USER_TEMPLATE = """Generate {n} multiple-choice questions from the following source text.

Difficulty level: {difficulty}
Cognitive level (Bloom's Taxonomy): {cognitive_level}

Cognitive level guidance:
- recall: ask about definitions, facts, or terms directly stated in the text
- comprehension: ask the student to explain or paraphrase a concept from the text
- application: ask the student to apply a concept from the text to a scenario
- analysis: ask the student to compare, contrast, or break down ideas in the text

Each question must have exactly 4 options (A, B, C, D) with one correct answer.
The distractors should be plausible but clearly wrong based on the source text.

Source text:
\"\"\"
{chunk}
\"\"\"

Respond with a JSON array. Each element must have these exact keys:
  "question": string
  "options": {{"A": string, "B": string, "C": string, "D": string}}
  "answer": "A" | "B" | "C" | "D"
  "difficulty": "{difficulty}"
  "cognitive_level": "{cognitive_level}"

JSON array:"""


def generate_candidates(
    chunk: str,
    difficulty: str = "medium",
    cognitive_level: str = "recall",
    n: int = 3,
) -> list[Question]:
    """
    Generate N candidate questions from a single text chunk.

    Args:
        chunk: Preprocessed source text.
        difficulty: "easy", "medium", or "hard".
        cognitive_level: "recall", "comprehension", "application", or "analysis".
        n: Number of candidate questions to generate.

    Returns:
        List of Question objects (may be fewer than N if parsing fails for some).
    """
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Invalid difficulty '{difficulty}'")
    if cognitive_level not in ("recall", "comprehension", "application", "analysis"):
        raise ValueError(f"Invalid cognitive_level '{cognitive_level}'")

    prompt = _USER_TEMPLATE.format(
        n=n,
        difficulty=difficulty,
        cognitive_level=cognitive_level,
        chunk=chunk,
    )

    raw = complete(_SYSTEM_PROMPT, prompt, max_tokens=1024 + 256 * n)
    return _parse_response(raw, chunk, difficulty, cognitive_level)


def generate_all(
    chunks: list[str],
    difficulty: str = "medium",
    cognitive_level: str = "recall",
    n: int = 3,
) -> list[list[Question]]:
    """
    Generate candidate questions for every chunk sequentially.
    Returns a list of lists — one inner list of candidates per chunk.
    """
    return [generate_candidates(chunk, difficulty, cognitive_level, n) for chunk in chunks]


def _parse_response(
    raw: str,
    chunk: str,
    difficulty: str,
    cognitive_level: str,
) -> list[Question]:
    """Parse the LLM JSON response into Question objects, tolerating minor formatting issues."""
    # Strip markdown code fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting the first JSON array from the response
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    questions = []
    for item in items:
        try:
            q = Question(
                question=item["question"],
                options=item["options"],
                answer=item["answer"].upper(),
                difficulty=item.get("difficulty", difficulty),
                cognitive_level=item.get("cognitive_level", cognitive_level),
                source_chunk=chunk,
            )
            questions.append(q)
        except (KeyError, AttributeError):
            continue

    return questions
