"""
Determinism regression check.

Runs tasks_edge_cases.json twice and diffs the answers.
Any task_id whose answer differs between runs is flagged loudly.

Usage (from project root):
    set LOCAL_DEV=true&& python tests/test_determinism.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
if os.environ.get("LOCAL_DEV"):
    load_dotenv(ROOT / ".env")

from src.config import load_config
from src.model_router import init_router, get_model_for_category
from src.classifier import classify_task
from src.fireworks_client import call_fireworks
from src.validator import validate_and_finalize, extract_last_resort
from src.prompts import get_prompt_builder, extract_math_answer, extract_code

TASKS_FILE = ROOT / "tests" / "sample_tasks" / "tasks_edge_cases.json"


def _post_process(category: str, raw: str) -> str:
    if category == "math_reasoning":
        return extract_math_answer(raw)
    if category == "code_generation":
        return extract_code(raw)
    return raw


async def _run_once(config, tasks: list) -> dict:
    results = {}
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"].strip()
        category = classify_task(prompt, task_id)
        model = get_model_for_category(category)
        messages = get_prompt_builder(category)(prompt)
        max_tokens = {
            "logical_reasoning": 2048, "math_reasoning": 1024,
            "code_generation": 1024, "code_debugging": 1024,
        }.get(category, 512)
        r = await call_fireworks(config, model, messages, category=category, max_tokens=max_tokens)
        raw = _post_process(category, r.content or "") if r.success else ""
        _, cleaned, _ = validate_and_finalize(task_id, category, raw, 0)
        results[task_id] = cleaned or (extract_last_resort(raw, category) if raw else "(no answer)")
    return results


async def main():
    config = load_config()
    init_router(config.allowed_models)
    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

    print(f"Running two passes over {len(tasks)} tasks from {TASKS_FILE.name} ...\n")
    run1 = await _run_once(config, tasks)
    run2 = await _run_once(config, tasks)

    diffs = {tid: (run1[tid], run2[tid]) for tid in run1 if run1[tid] != run2[tid]}

    if not diffs:
        print("✅  All answers identical across both runs — determinism check passed.")
        sys.exit(0)

    print(f"❌  {len(diffs)} task(s) returned different answers across runs:\n")
    for tid, (a1, a2) in diffs.items():
        print(f"  task_id: {tid}")
        print(f"    run1: {a1[:120]}")
        print(f"    run2: {a2[:120]}")
        print()
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
