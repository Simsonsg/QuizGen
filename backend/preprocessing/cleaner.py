"""
Rule-based cleaning of raw extracted text.

Removes common noise from lecture slides and notes:
- Slide/page numbers (e.g. "Slide 3", "Page 3 of 10", standalone digits)
- Short lines that are likely headers or footers
- Repeated whitespace and blank lines
- Common boilerplate phrases
"""

import re


_BOILERPLATE = re.compile(
    r"""
    ^\s*(
        slide\s*\d+           |   # "Slide 3"
        page\s*\d+(\s*of\s*\d+)?  |   # "Page 3 of 10"
        \d+\s*/\s*\d+         |   # "3/10"
        ^\d+\s*$              |   # standalone number
        confidential          |
        all\s+rights\s+reserved |
        www\.\S+              |   # URLs
        https?://\S+
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

_EXCESS_WHITESPACE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """
    Apply rule-based cleaning to raw extracted text.
    Returns cleaned text with noise removed.
    """
    text = raw

    # Remove boilerplate lines
    text = _BOILERPLATE.sub("", text)

    # Drop very short lines (< 4 chars) that are likely artefacts
    lines = text.splitlines()
    lines = [l for l in lines if len(l.strip()) >= 4 or l.strip() == ""]

    text = "\n".join(lines)

    # Collapse excess blank lines
    text = _EXCESS_WHITESPACE.sub("\n\n", text)

    return text.strip()
