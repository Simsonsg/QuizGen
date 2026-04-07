"""
Segment text into semantically coherent chunks using sentence boundaries.

Strategy:
- Split into sentences with NLTK
- Accumulate sentences until a word-count ceiling is reached
- Never break mid-sentence
"""

import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer data on first use
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def chunk_text(text: str, max_words: int = 200, min_words: int = 40) -> list[str]:
    """
    Split text into chunks bounded by sentence boundaries.

    Args:
        text: Cleaned input text.
        max_words: Maximum words per chunk before forcing a split.
        min_words: Minimum words for a chunk to be kept (drops trailing fragments).

    Returns:
        List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_words + word_count > max_words and current:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
            current = [sentence]
            current_words = word_count
        else:
            current.append(sentence)
            current_words += word_count

    # Flush remaining sentences
    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)

    return chunks
