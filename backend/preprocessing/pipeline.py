"""
Preprocessing pipeline entry point.
Supports three strategies as described in the system design:
  - "raw"      : parse only, no cleaning
  - "clean"    : parse + rule-based cleaning
  - "summarise": parse + rule-based cleaning + LLM summarisation
"""

from .parser import parse_file
from .cleaner import clean_text
from .chunker import chunk_text
from .summariser import summarise_chunks

Strategy = str  # "raw" | "clean" | "summarise"


def preprocess(
    file_path: str,
    strategy: Strategy = "clean",
    max_words: int = 200,
    min_words: int = 40,
) -> list[str]:
    """
    Run the preprocessing pipeline on a document file.

    Args:
        file_path: Path to the input document (.pdf, .pptx, or .txt).
        strategy: Preprocessing strategy — "raw", "clean", or "summarise".
        max_words: Maximum words per chunk.
        min_words: Minimum words for a chunk to be retained.

    Returns:
        List of text chunks ready for question generation.
    """
    if strategy not in ("raw", "clean", "summarise"):
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: raw, clean, summarise")

    raw_text = parse_file(file_path)

    if strategy == "raw":
        text = raw_text
    else:
        text = clean_text(raw_text)

    chunks = chunk_text(text, max_words=max_words, min_words=min_words)

    if strategy == "summarise":
        chunks = summarise_chunks(chunks)

    return chunks
