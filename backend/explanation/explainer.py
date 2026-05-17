"""
Grounded explanation generation.

For each validated question, the LLM generates an explanation that:
  - justifies why the correct answer is right
  - clarifies why each distractor is wrong
  - is strictly grounded in the source chunk (no external knowledge)

Explanations are then checked for semantic alignment with the source.
Those with low alignment are discarded.
"""

from backend.llm import complete
from backend.generation.question import Question
from backend.validation.scorer import similarity

_SYSTEM_PROMPT = (
    "You are an expert tutor explaining multiple-choice questions to students. "
    "Your explanations are grounded strictly in the provided source text. "
    "Do not introduce facts, definitions, or claims not present in the source. "
    "Be concise and clear."
)

_USER_TEMPLATE = """Source text:
\"\"\"
{chunk}
\"\"\"

Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Correct answer: {answer}) {correct_text}

Using only the source text above, write a short explanation (3–5 sentences) that:
1. Explains why the correct answer is right, citing or paraphrasing the source.
2. Briefly explains why each of the other options is incorrect.

Explanation:"""


def generate_explanation(question: Question) -> str:
    """
    Generate a grounded explanation for a validated question.
    Returns the explanation string.
    """
    prompt = _USER_TEMPLATE.format(
        chunk=question.source_chunk,
        question=question.question,
        opt_a=question.options["A"],
        opt_b=question.options["B"],
        opt_c=question.options["C"],
        opt_d=question.options["D"],
        answer=question.answer,
        correct_text=question.correct_answer_text(),
    )
    return complete(_SYSTEM_PROMPT, prompt, max_tokens=400)


def generate_and_attach(
    questions: list[Question],
    alignment_threshold: float = 0.20,
) -> list[Question]:
    """
    Generate explanations for all questions, validate alignment with source,
    and attach passing explanations in-place. Returns the same list.
    """
    for q in questions:
        explanation = generate_explanation(q)
        score = similarity(explanation, q.source_chunk)
        q.explanation = explanation if score >= alignment_threshold else ""
    return questions
