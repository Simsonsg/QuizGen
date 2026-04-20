"""
Cognitive-level validation.

Asks the LLM to independently classify the Bloom's Taxonomy level of a generated
question, then checks whether it matches the intended level.

This measures how well difficulty conditioning actually works — useful evaluation
data for the dissertation.

Bloom's levels used in this system:
  recall        — remembering facts, definitions, terms
  comprehension — explaining or paraphrasing concepts
  application   — applying a concept to a scenario
  analysis      — comparing, contrasting, or breaking down ideas
"""

from backend.llm import complete

_SYSTEM_PROMPT = (
    "You are an expert in educational assessment and Bloom's Taxonomy. "
    "Classify the cognitive level of the given question using exactly one of these labels: "
    "recall, comprehension, application, analysis. "
    "Reply with the label only — no explanation, no punctuation."
)

_USER_TEMPLATE = """Classify the Bloom's Taxonomy level of this multiple-choice question.

Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Reply with exactly one word: recall | comprehension | application | analysis"""

_VALID_LEVELS = {"recall", "comprehension", "application", "analysis"}


def classify_cognitive_level(question: str, options: dict[str, str]) -> str | None:
    """
    Ask the LLM to classify the cognitive level of a question.
    Returns one of: recall, comprehension, application, analysis.
    Returns None if the response is unrecognised.
    """
    prompt = _USER_TEMPLATE.format(
        question=question,
        opt_a=options.get("A", ""),
        opt_b=options.get("B", ""),
        opt_c=options.get("C", ""),
        opt_d=options.get("D", ""),
    )
    response = complete(_SYSTEM_PROMPT, prompt, max_tokens=10).strip().lower()

    for level in _VALID_LEVELS:
        if level in response:
            return level
    return None


def check_cognitive_level(question_text: str, options: dict[str, str], intended_level: str) -> tuple[bool, str | None]:
    """
    Check whether the actual cognitive level matches the intended level.

    Returns:
        (matches: bool, actual_level: str | None)
    """
    actual = classify_cognitive_level(question_text, options)
    matches = actual == intended_level
    return matches, actual
