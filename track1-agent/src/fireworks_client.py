import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import httpx

from src.config import Config

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_TIMEOUT = 25.0
MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 2.0]


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

    last_result = CallResult(success=False, error="No attempts made")

    for attempt in range(MAX_RETRIES + 1):
        if deadline is not None and time.monotonic() >= deadline - 2.0:
            logger.warning("Deadline too close — skipping attempt %d for category %r", attempt, category)
            return CallResult(success=False, error="Deadline exceeded before attempt")

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
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

            elif resp.status_code == 429 or resp.status_code >= 500:
                last_result = CallResult(success=False, error=f"HTTP {resp.status_code}")
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning("HTTP %d — retrying in %.1fs", resp.status_code, delay)
                    await asyncio.sleep(delay)
                continue
            else:
                # 4xx non-429: don't retry
                last_result = CallResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
                logger.error("Non-retryable HTTP %d for category=%s", resp.status_code, category)
                return last_result

        except httpx.TimeoutException:
            latency = time.monotonic() - t0
            last_result = CallResult(success=False, error="Timeout")
            logger.warning("Timeout after %.2fs for category=%s attempt=%d", latency, category, attempt)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAYS[attempt])

        except httpx.RequestError as e:
            last_result = CallResult(success=False, error=f"Network error: {e}")
            logger.error("Network error for category=%s: %s", category, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAYS[attempt])

    return last_result
