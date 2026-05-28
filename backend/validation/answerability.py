"""
LLM-based answerability check.
"""

from backend.llm import complete

_SYSTEM_PROMPT = (
    "You are a strict educational content reviewer. "
    "Answer only with 'YES' or 'NO'. Do not explain."
)

_USER_TEMPLATE = """Source text:
\"\"\"
{chunk}
\"\"\"

Question: {question}
Claimed correct answer: {answer_text}

Can the correct answer be directly supported or inferred from the source text above?
Reply YES or NO only."""


def is_answerable(question: str, answer_text: str, chunk: str) -> bool:
    """
    Returns True if the correct answer is supported by the source chunk.
    """
    prompt = _USER_TEMPLATE.format(
        chunk=chunk,
        question=question,
        answer_text=answer_text,
    )
    response = complete(_SYSTEM_PROMPT, prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")
