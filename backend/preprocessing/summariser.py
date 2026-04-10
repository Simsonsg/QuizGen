"""
LLM-based summarisation of text chunks.

Uses the shared backend.llm module — provider is configured via LLM_PROVIDER in .env.
"""

from backend.llm import complete

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


def summarise_chunk(chunk: str) -> str:
    return complete(_SYSTEM_PROMPT, _USER_TEMPLATE.format(chunk=chunk), max_tokens=512)


def summarise_chunks(chunks: list[str]) -> list[str]:
    return [summarise_chunk(chunk) for chunk in chunks]
