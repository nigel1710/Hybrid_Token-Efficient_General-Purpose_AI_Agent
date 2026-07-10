import logging
import os
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_total_prompt_tokens = 0
_total_completion_tokens = 0
_per_task: List[Dict[str, Any]] = []

TOKEN_LOG_PATH = os.environ.get("TOKEN_LOG_PATH", "tests/token_usage.txt")


def record(usage: dict, task_id: str = "", category: str = "", model: str = ""):
    global _total_prompt_tokens, _total_completion_tokens
    if not usage:
        return
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    _total_prompt_tokens += pt
    _total_completion_tokens += ct
    _per_task.append({
        "task_id": task_id,
        "category": category,
        "model": model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    })


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


def write_token_log(tasks_file: str = ""):
    """Write a human-readable per-task token breakdown to TOKEN_LOG_PATH."""
    s = summary()
    lines = [
        "Token Usage Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tasks file: {tasks_file}",
        "=" * 70,
        f"{'Task ID':<12} {'Category':<26} {'Model':<25} {'Prompt':>7} {'Compl':>7} {'Total':>7}",
        "-" * 70,
    ]
    for t in _per_task:
        short_model = t["model"].split("/")[-1] if "/" in t["model"] else t["model"]
        lines.append(
            f"{t['task_id']:<12} {t['category']:<26} {short_model:<25} "
            f"{t['prompt_tokens']:>7} {t['completion_tokens']:>7} {t['total_tokens']:>7}"
        )
    lines += [
        "-" * 70,
        f"{'TOTAL':<12} {'':<26} {'':<25} "
        f"{s['prompt_tokens']:>7} {s['completion_tokens']:>7} {s['total_tokens']:>7}",
        "=" * 70,
    ]
    os.makedirs(os.path.dirname(TOKEN_LOG_PATH) or ".", exist_ok=True)
    with open(TOKEN_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Token log written to %s", TOKEN_LOG_PATH)
