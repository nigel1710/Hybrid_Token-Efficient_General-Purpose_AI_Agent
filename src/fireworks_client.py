import asyncio
import logging
import time
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


async def call_fireworks(
    config: Config,
    model: str,
    messages: list,
    category: str = "",
    deadline: Optional[float] = None,
    max_tokens: int = 1024,
) -> CallResult:
    extra_body: Dict[str, Any] = {}
    if "kimi-k2p7-code" in model.lower():
        # Disable visible thinking for Kimi model by default to save tokens and avoid leaks
        extra_body["reasoning_effort"] = "none"

    gemma = _is_gemma(model)
    timeout = GEMMA_TIMEOUT if gemma else DEFAULT_TIMEOUT
    retry_delays = GEMMA_RETRY_DELAYS if gemma else RETRY_DELAYS
    max_retries = GEMMA_MAX_RETRIES if gemma else MAX_RETRIES

    client = _get_client(config, timeout)

    last_result = CallResult(success=False, error="No attempts made")

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
                max_tokens=max_tokens,
                temperature=0,
                top_p=0.1,
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
