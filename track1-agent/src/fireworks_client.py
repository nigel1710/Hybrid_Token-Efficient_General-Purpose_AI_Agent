import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import httpx

from src.config import Config

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Standard (non-Gemma) settings
DEFAULT_TIMEOUT = 25.0
MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 2.0]

# Gemma on-demand deployment settings — 404 means cold-starting, not a real error
GEMMA_TIMEOUT = 28.0
GEMMA_RETRY_DELAYS = [3.0, 6.0, 10.0]
GEMMA_MAX_RETRIES = 3  # one extra attempt to absorb spin-up time


def _is_gemma(model: str) -> bool:
    return "gemma" in model.lower()


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
    url = config.base_url + CHAT_COMPLETIONS_PATH
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if "kimi-k2p7-code" in model.lower():
        # Disable visible thinking for Kimi model by default to save tokens and avoid leaks
        payload["reasoning_effort"] = "none"

    gemma = _is_gemma(model)
    timeout = GEMMA_TIMEOUT if gemma else DEFAULT_TIMEOUT
    retry_delays = GEMMA_RETRY_DELAYS if gemma else RETRY_DELAYS
    max_retries = GEMMA_MAX_RETRIES if gemma else MAX_RETRIES

    last_result = CallResult(success=False, error="No attempts made")

    for attempt in range(max_retries + 1):
        if deadline is not None and time.monotonic() >= deadline - 2.0:
            logger.warning("Deadline too close — skipping attempt %d for category=%s model=%s",
                           attempt, category, model)
            return CallResult(success=False, error="Deadline exceeded before attempt")

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)

            latency = time.monotonic() - t0
            logger.info("category=%s model=%s attempt=%d status=%d latency=%.2fs",
                        category, model, attempt, resp.status_code, latency)

            if resp.status_code == 200:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    last_result = CallResult(success=False, error="Unexpected response schema")
                    logger.warning("Unexpected schema from Fireworks: %s", str(data)[:200])
                    continue

                if not content or not content.strip():
                    last_result = CallResult(success=False, error="Empty completion returned")
                    logger.warning("Empty completion for category=%s attempt=%d", category, attempt)
                    continue

                usage = data.get("usage")
                logger.info("usage=%s", usage)
                return CallResult(success=True, content=content.strip(), usage=usage)

            elif resp.status_code == 404 and gemma:
                # Gemma on-demand: 404 = scaled to zero / cold-starting — retryable
                last_result = CallResult(success=False, error="HTTP 404 (Gemma cold-start)")
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning("Gemma model %s returned 404 (cold-start/scale-to-zero) — "
                                   "retrying in %.0fs (attempt %d/%d)",
                                   model, delay, attempt + 1, max_retries)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Gemma model %s failed to warm up after %d attempts — "
                                 "deployment may be unavailable", model, max_retries + 1)
                continue

            elif resp.status_code == 429 or resp.status_code >= 500:
                last_result = CallResult(success=False, error=f"HTTP {resp.status_code}")
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning("HTTP %d — retrying in %.1fs", resp.status_code, delay)
                    await asyncio.sleep(delay)
                continue

            else:
                # Non-retryable: 4xx that is not 429, and not a Gemma cold-start 404
                last_result = CallResult(
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                logger.error("Non-retryable HTTP %d for category=%s model=%s",
                             resp.status_code, category, model)
                return last_result

        except httpx.TimeoutException:
            latency = time.monotonic() - t0
            last_result = CallResult(success=False, error="Timeout")
            logger.warning("Timeout after %.2fs for category=%s model=%s attempt=%d",
                           latency, category, model, attempt)
            if attempt < max_retries:
                await asyncio.sleep(retry_delays[attempt])

        except httpx.RequestError as e:
            last_result = CallResult(success=False, error=f"Network error: {e}")
            logger.error("Network error for category=%s model=%s: %s", category, model, e)
            if attempt < max_retries:
                await asyncio.sleep(retry_delays[attempt])

    return last_result
