"""
Unified LLM client.

Configured via environment variables:
  LLM_PROVIDER = anthropic | groq | gemini   (default: groq)
  LLM_MODEL    = model name override          (optional)
  ANTHROPIC_API_KEY
  GROQ_API_KEY
  GEMINI_API_KEY

Groq and Gemini use OpenAI-compatible endpoints via the openai package.
Anthropic uses the anthropic package directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Default models per provider
_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-1.5-flash",
}

_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_default_model() -> str:
    return os.getenv("LLM_MODEL", _DEFAULTS[PROVIDER])


def complete(system: str, user: str, model: str | None = None, max_tokens: int = 512) -> str:
    """
    Send a system + user message and return the assistant's reply as a string.
    Works across all supported providers.
    """
    model = model or get_default_model()

    if PROVIDER == "anthropic":
        return _complete_anthropic(system, user, model, max_tokens)
    elif PROVIDER in ("groq", "gemini"):
        return _complete_openai_compat(system, user, model, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Choose from: anthropic, groq, gemini")


def _complete_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text.strip()


def _complete_openai_compat(system: str, user: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI
    api_key = os.getenv(_API_KEY_ENV[PROVIDER])
    client = OpenAI(api_key=api_key, base_url=_BASE_URLS[PROVIDER])
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()
