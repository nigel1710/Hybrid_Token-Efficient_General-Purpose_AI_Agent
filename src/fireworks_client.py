import asyncio
import logging
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

# Standard (non-Gemma) settings
DEFAULT_TIMEOUT = 25.0
MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 2.0]

# Gemma on-demand deployment settings — 404 means cold-starting, not a real error
GEMMA_TIMEOUT = 28.0
GEMMA_RETRY_DELAYS = [3.0, 6.0, 10.0]
GEMMA_MAX_RETRIES = 3  # one extra attempt to absorb spin-up time

# Module-level OpenAI client cache keyed by (api_key, base_url)
_client_cache: Dict[tuple, OpenAI] = {}


def _is_gemma(model: str) -> bool:
    return "gemma" in model.lower()


def _get_client(config: Config, timeout: float) -> OpenAI:
    key = (config.api_key, config.base_url, timeout)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout,
        )
    return _client_cache[key]


@dataclass
class CallResult:
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


# Category-specific maximum token limits (tight but safe caps)
CATEGORY_MAX_TOKENS = {
    "factual_knowledge": 130,
    "math_reasoning": 220,
    "sentiment_classification": 80,
    "summarisation": 110,
    "ner": 150,
    "code_debugging": 160,
    "logical_reasoning": 220,
    "code_generation": 160,
}

# Extra budget for factual questions that compare 2+ named concepts —
# these need more room than simple single-fact questions, since they must
# cover multiple distinct dimensions (e.g. volatility, speed, use case)
# without being truncated mid-explanation.
FACTUAL_COMPARISON_MAX_TOKENS = 260

_COMPARISON_PATTERN = re.compile(
    r"\bdifference between\b|\bcompare\b|\bcompared to\b|\bversus\b|\bvs\.?\b",
    re.IGNORECASE,
)

# Category-appropriate stop sequences
CATEGORY_STOP_SEQUENCES = {
    "ner": ["}"],
    "code_generation": ["\n```\n", "\n```\r\n"],
    "code_debugging": ["\n```\n", "\n```\r\n"],
    "math_reasoning": ["[END]"],
}


async def call_fireworks(
    config: Config,
    model: str,
    messages: list,
    category: str = "",
    deadline: Optional[float] = None,
    max_tokens: int = 1024,
) -> CallResult:
    extra_body: Dict[str, Any] = {}
    if "kimi" in model.lower():
        # Disable visible thinking for Kimi model by default to save tokens and avoid leaks
        extra_body["reasoning_effort"] = "none"
    elif "minimax" in model.lower():
        if category == "logical_reasoning":
            extra_body["thinking"] = {"type": "enabled"}
        else:
            extra_body["thinking"] = {"type": "disabled"}

    gemma = _is_gemma(model)
    timeout = GEMMA_TIMEOUT if gemma else DEFAULT_TIMEOUT
    retry_delays = GEMMA_RETRY_DELAYS if gemma else RETRY_DELAYS
    max_retries = GEMMA_MAX_RETRIES if gemma else MAX_RETRIES

    client = _get_client(config, timeout)

    last_result = CallResult(success=False, error="No attempts made")

    user_msg = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_msg = msg.get("content", "")

    # Override for summarisation tasks misclassified as math_reasoning or factual_knowledge
    is_summary = False
    if user_msg:
        if bool(re.search(
            r"\bsummar[iy]s[ei]\b|\bsummar[iy]z[ei]\b|\bcondense\b|\btl;?dr\b|\bshorten\b|"
            r"in one sentence|in \w+ (bullet|sentence|word)|under \w+ word|bullet point",
            user_msg, re.IGNORECASE
        )):
            is_summary = True

    # Detect comparison-style factual questions (e.g. "difference between RAM and ROM")
    # which need more completion budget than simple single-fact questions to avoid
    # truncating mid-explanation or dropping a required comparison dimension.
    is_comparison = (
        category == "factual_knowledge"
        and not is_summary
        and bool(user_msg and _COMPARISON_PATTERN.search(user_msg))
    )

    if is_summary:
        resolved_max_tokens = CATEGORY_MAX_TOKENS.get("summarisation", 110)
        stop_seq = CATEGORY_STOP_SEQUENCES.get("summarisation")
    elif is_comparison:
        resolved_max_tokens = FACTUAL_COMPARISON_MAX_TOKENS
        stop_seq = CATEGORY_STOP_SEQUENCES.get(category)
    else:
        resolved_max_tokens = CATEGORY_MAX_TOKENS.get(category, max_tokens)
        stop_seq = CATEGORY_STOP_SEQUENCES.get(category)

    if category == "logical_reasoning":
        temperature = 0.2
        top_p = 0.9
    else:
        temperature = 0
        top_p = 0.1

    for attempt in range(max_retries + 1):
        if deadline is not None and time.monotonic() >= deadline - 2.0:
            logger.warning(
                "Deadline too close — skipping attempt %d for category=%s model=%s",
                attempt, category, model,
            )
            return CallResult(success=False, error="Deadline exceeded before attempt")

        t0 = time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=resolved_max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop_seq,
                extra_body=extra_body if extra_body else None,
            )

            latency = time.monotonic() - t0
            logger.info(
                "category=%s model=%s attempt=%d latency=%.2fs",
                category, model, attempt, latency,
            )

            content = response.choices[0].message.content
            if content is None:
                content = ""

            # Stop sequence reconstruction
            if category == "ner" and content.strip() and not content.strip().endswith("}"):
                content = content.strip() + "}"
            elif category in ("code_generation", "code_debugging") and content.strip():
                if "```" in content and content.count("```") % 2 != 0:
                    content = content.rstrip() + "\n```"

            if not content.strip():
                last_result = CallResult(success=False, error="Empty completion returned")
                logger.warning("Empty completion for category=%s attempt=%d", category, attempt)
                if attempt < max_retries:
                    await asyncio.sleep(retry_delays[attempt])
                continue

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            logger.info("usage=%s", usage)
            return CallResult(success=True, content=content.strip(), usage=usage)

        except Exception as exc:
            latency = time.monotonic() - t0
            exc_str = str(exc)

            # Map OpenAI SDK HTTP errors to the same retry behaviour as before
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is None:
                # Try to pull it from the exception body (openai.APIStatusError)
                status_code = getattr(exc, "status_code", None)

            if status_code == 404 and gemma:
                last_result = CallResult(success=False, error="HTTP 404 (Gemma cold-start)")
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning(
                        "Gemma model %s returned 404 (cold-start/scale-to-zero) — "
                        "retrying in %.0fs (attempt %d/%d)",
                        model, delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Gemma model %s failed to warm up after %d attempts — "
                        "deployment may be unavailable", model, max_retries + 1,
                    )
                continue

            if status_code == 429 or (status_code is not None and status_code >= 500):
                last_result = CallResult(success=False, error=f"HTTP {status_code}")
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning("HTTP %d — retrying in %.1fs", status_code, delay)
                    await asyncio.sleep(delay)
                continue

            if status_code is not None:
                # Non-retryable 4xx
                last_result = CallResult(success=False, error=f"HTTP {status_code}: {exc_str[:200]}")
                logger.error(
                    "Non-retryable HTTP %d for category=%s model=%s",
                    status_code, category, model,
                )
                return last_result

            # Network / timeout errors — retryable
            last_result = CallResult(success=False, error=f"Request error: {exc_str[:200]}")
            logger.warning(
                "Request error after %.2fs for category=%s model=%s attempt=%d: %s",
                latency, category, model, attempt, exc_str,
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delays[attempt])

    return last_result