import logging

logger = logging.getLogger(__name__)

_total_prompt_tokens = 0
_total_completion_tokens = 0


def record(usage: dict):
    global _total_prompt_tokens, _total_completion_tokens
    if not usage:
        return
    _total_prompt_tokens += usage.get("prompt_tokens", 0)
    _total_completion_tokens += usage.get("completion_tokens", 0)


def summary() -> dict:
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
    }


def log_summary():
    s = summary()
    logger.info("Token usage — prompt: %d  completion: %d  total: %d",
                s["prompt_tokens"], s["completion_tokens"], s["total_tokens"])
