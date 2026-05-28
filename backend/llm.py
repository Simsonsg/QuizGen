"""
LLM provider 
"""

import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Build provider list from LLM_PROVIDERS (comma-separated) or fall back to LLM_PROVIDER
_raw = os.getenv("LLM_PROVIDERS") or os.getenv("LLM_PROVIDER", "groq")
PROVIDERS: list[str] = [p.strip().lower() for p in _raw.split(",") if p.strip()]

_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.3-70b-versatile",
    "gemini":    "gemini-2.0-flash",
}

_BASE_URLS = {
    "groq":   "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}

# Transient errors worth retrying (per attempt, within a single provider)
_MAX_RETRIES = 2
_RETRY_BACKOFF = [1, 3]  # seconds to wait before attempt 2 and 3


def _model_for(provider: str) -> str:
    env_key = f"LLM_MODEL_{provider.upper()}"
    return os.getenv(env_key) or os.getenv("LLM_MODEL") or _DEFAULTS[provider]


def complete(system: str, user: str, model: str | None = None, max_tokens: int = 512) -> str:
    """
    Send a system + user message and return the assistant reply.
    """
    last_error: Exception | None = None

    for provider in PROVIDERS:
        resolved_model = model or _model_for(provider)

        for attempt in range(_MAX_RETRIES + 1):
            try:
                logger.debug("Trying provider=%s model=%s attempt=%d", provider, resolved_model, attempt + 1)
                result = _call(provider, system, user, resolved_model, max_tokens)
                if provider != PROVIDERS[0]:
                    logger.info("Succeeded on fallback provider: %s", provider)
                return result

            except Exception as e:
                last_error = e
                if _is_rate_limit(e):
                    logger.warning("Rate limit hit on %s — moving to next provider.", provider)
                    break  # don't retry this provider
                elif _is_transient(e) and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning("Transient error on %s (attempt %d), retrying in %ds: %s", provider, attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    logger.warning("Non-retryable error on %s: %s", provider, e)
                    break  # move to next provider

    raise RuntimeError(
        f"All providers exhausted ({', '.join(PROVIDERS)}). Last error: {last_error}"
    ) from last_error



def _call(provider: str, system: str, user: str, model: str, max_tokens: int) -> str:
    if provider == "anthropic":
        return _call_anthropic(system, user, model, max_tokens)
    elif provider in ("groq", "gemini"):
        return _call_openai_compat(provider, system, user, model, max_tokens)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: anthropic, groq, gemini")


def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text.strip()


def _call_openai_compat(provider: str, system: str, user: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv(_API_KEY_ENV[provider]), base_url=_BASE_URLS[provider])
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _is_rate_limit(e: Exception) -> bool:
    try:
        from openai import RateLimitError as OAIRateLimit
        if isinstance(e, OAIRateLimit):
            return True
    except ImportError:
        pass
    try:
        from anthropic import RateLimitError as ANTRateLimit
        if isinstance(e, ANTRateLimit):
            return True
    except ImportError:
        pass
    return False


def _is_transient(e: Exception) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(e, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    try:
        from anthropic import APIConnectionError, APITimeoutError
        if isinstance(e, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    return False
