"""
Known-answer regression checks.

Runs specific tasks whose correct answers are ground-truth verified and asserts
the pipeline produces the right answer. Any failure is an immediate regression signal.

Ground-truth answers confirmed by hand:
  a21 — Carol/season puzzle  → "Summer"
  e05 — Alice/Bob/Carol/Dave seating puzzle → "Alice"

Usage (from project root):
    set LOCAL_DEV=true&& python tests/test_known_answers.py
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
from src.validator import validate_and_finalize, extract_last_resort, strip_reasoning_blocks
from src.prompts import get_prompt_builder, extract_math_answer, extract_code

# Ground-truth: task_id -> substring that must appear in the answer (case-insensitive)
KNOWN_ANSWERS: dict[str, str] = {
    "a21": "summer",
    "e05": "alice",
}

ALL_TASKS_FILES = [
    ROOT / "tests" / "sample_tasks" / "tasks_all_categories.json",
    ROOT / "tests" / "sample_tasks" / "tasks_edge_cases.json",
]


def _post_process(category: str, raw: str) -> str:
    if category == "math_reasoning":
        return extract_math_answer(raw)
    if category == "code_generation":
        return extract_code(raw)
    return raw


async def _run_task(config, task: dict) -> str:
    task_id = task["task_id"]
    prompt = task["prompt"].strip()
    category = classify_task(prompt, task_id)
    model = get_model_for_category(category)
    messages = get_prompt_builder(category)(prompt)
    max_tokens = {
        "logical_reasoning": 2048, "math_reasoning": 1024,
        "code_generation": 1024, "code_debugging": 1024,
        "compound": 1024,
    }.get(category, 512)

    # For logical_reasoning run two calls and pick the consistent answer
    if category == "logical_reasoning":
        async def _call():
            r = await call_fireworks(config, model, messages, category=category, max_tokens=max_tokens)
            return strip_reasoning_blocks(r.content).strip() if r.success and r.content else None

        a1, a2 = await asyncio.gather(_call(), _call())
        if a1 and a2 and strip_reasoning_blocks(a1).lower() == strip_reasoning_blocks(a2).lower():
            return a1
        # disagreement — use tiebreak
        tb_messages = get_prompt_builder(category)(
            prompt + "\n\n[Carefully re-check every constraint one at a time before giving your final answer.]"
        )
        r3 = await call_fireworks(config, model, tb_messages, category=category, max_tokens=max_tokens)
        return (r3.content or a1 or "").strip()

    r = await call_fireworks(config, model, messages, category=category, max_tokens=max_tokens)
    raw = _post_process(category, r.content or "") if r.success else ""
    _, cleaned, _ = validate_and_finalize(task_id, category, raw, 0)
    return cleaned or (extract_last_resort(raw, category) if raw else "(no answer)")


async def main():
    config = load_config()
    init_router(config.allowed_models)

    # Collect the target tasks from whichever file contains them
    target_tasks: dict[str, dict] = {}
    for path in ALL_TASKS_FILES:
        if not path.exists():
            continue
        for task in json.loads(path.read_text(encoding="utf-8")):
            if task["task_id"] in KNOWN_ANSWERS:
                target_tasks[task["task_id"]] = task

    missing = set(KNOWN_ANSWERS) - set(target_tasks)
    if missing:
        print(f"WARNING: task_id(s) {missing} not found in any sample tasks file — skipping.")

    if not target_tasks:
        print("No known-answer tasks found. Exiting.")
        sys.exit(0)

    print(f"Running {len(target_tasks)} known-answer task(s): {list(target_tasks)}\n")

    failures = []
    for task_id, task in target_tasks.items():
        answer = await _run_task(config, task)
        expected = KNOWN_ANSWERS[task_id]
        passed = expected.lower() in answer.lower()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  task_id={task_id}  expected={expected!r}  got={answer[:120]!r}")
        if not passed:
            failures.append(task_id)

    print()
    if failures:
        print(f"❌  {len(failures)} known-answer regression(s) failed: {failures}")
        sys.exit(1)
    else:
        print("✅  All known-answer checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
