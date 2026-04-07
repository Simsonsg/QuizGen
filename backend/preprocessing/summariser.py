"""
LLM-based summarisation of text chunks using the Claude API.
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

_CLIENT = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


_SYSTEM_PROMPT = (
    "You are an expert at condensing educational content. "
    "Your summaries are concise, factually accurate, and retain all key concepts, "
    "definitions, and relationships from the source material. "
    "Do not add information not present in the source."
)

_USER_TEMPLATE = (
    "Summarise the following educational content into a single coherent paragraph. "
    "Preserve all key concepts, terms, and factual claims. "
    "Do not include meta-commentary about the summary itself.\n\n"
    "Content:\n{chunk}"
)


def summarise_chunk(chunk: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """
    Summarise a single text chunk using Claude.

    Uses Haiku by default for cost-efficiency during preprocessing.
    """
    client = _get_client()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _USER_TEMPLATE.format(chunk=chunk)}],
    )
    return message.content[0].text.strip()


def summarise_chunks(chunks: list[str], model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """
    Summarise a list of text chunks. Returns a list of summary strings
    in the same order as the input.
    """
    return [summarise_chunk(chunk, model=model) for chunk in chunks]
