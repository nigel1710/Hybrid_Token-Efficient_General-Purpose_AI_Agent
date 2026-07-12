import asyncio
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Set

from logger_setup import setup_logging
setup_logging()

from config import load_config
from io_handler import read_tasks, write_results
from classifier import classify_task
from model_router import init_router, get_model_for_category, mark_model_unavailable
from fireworks_client import call_fireworks, _is_gemma
from validator import validate_and_finalize, extract_last_resort, strip_reasoning_blocks, check_min_completeness
import token_tracker as token_tracker
from prompts import get_prompt_builder, extract_math_answer, extract_code

logger = logging.getLogger(__name__)

DEADLINE_SECONDS = 9.5 * 60       # task-processing budget (starts AFTER warm-up)
WARMUP_BUDGET_SECONDS = 45.0      # max time to spend warming up Gemma models
CONCURRENCY_LIMIT = 8
MAX_TOKENS_BY_CATEGORY = {
    "logical_reasoning": 220,
    "math_reasoning": 70,
    "code_generation": 160,
    "code_debugging": 160,
    "summarisation": 110,
    "ner": 150,
    "factual_knowledge": 130,
    "sentiment_classification": 80,
    "compound": 1024,
}
MAX_PROMPT_CHARS = 12000  # ~3k tokens; truncate if longer

# Minimal warm-up prompt — just enough to trigger model spin-up
_WARMUP_MESSAGES = [{"role": "user", "content": "Hi"}]


def _build_messages(category: str, prompt: str) -> list:
    return get_prompt_builder(category)(prompt)


def _post_process(category: str, raw: str) -> str:
    if category == "math_reasoning":
        return extract_math_answer(raw)
    if category == "code_generation":
        return extract_code(raw)
    return raw


def _truncate_prompt(prompt: str) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    logger.warning("Prompt truncated from %d to %d chars", len(prompt), MAX_PROMPT_CHARS)
    head = int(MAX_PROMPT_CHARS * 0.2)
    tail = MAX_PROMPT_CHARS - head
    return prompt[:head] + "\n...[truncated]...\n" + prompt[-tail:]


async def _warmup_gemma_models(config, allowed_models: List[str]) -> Set[str]:
    """
    Send a lightweight request to each Gemma model to trigger cold-start spin-up
    before the main task-processing clock starts. Returns the set of model IDs
    that failed to warm up (so the router can fall back for this run).
    Time budget: WARMUP_BUDGET_SECONDS total across all Gemma models.
    """
    gemma_models = [m for m in allowed_models if _is_gemma(m)]
    if not gemma_models:
        return set()

    logger.info("Starting Gemma warm-up for %d model(s): %s", len(gemma_models), gemma_models)
    warmup_deadline = time.monotonic() + WARMUP_BUDGET_SECONDS
    failed: Set[str] = set()

    async def warmup_one(model: str):
        result = await call_fireworks(
            config, model, _WARMUP_MESSAGES,
            category="warmup",
            deadline=warmup_deadline,
            max_tokens=8,
        )
        if result.success:
            logger.info("Warm-up OK: %s", model)
        else:
            logger.warning(
                "Warm-up FAILED for %s (%s) — this model will be skipped for this run; "
                "falling back to next tier. This is a deployment availability issue, not a code bug.",
                model, result.error
            )
            failed.add(model)

    await asyncio.gather(*[warmup_one(m) for m in gemma_models])

    elapsed = time.monotonic() + WARMUP_BUDGET_SECONDS - warmup_deadline
    logger.info("Warm-up complete in %.1fs — %d/%d Gemma models ready",
                elapsed, len(gemma_models) - len(failed), len(gemma_models))
    return failed


async def _logical_reasoning_with_consistency(
    task_id: str, prompt: str, config, model: str, max_tokens: int,
    deadline: float, semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Two parallel calls; if they agree return immediately. If not, make a third tie-breaking call."""
    messages = _build_messages("logical_reasoning", prompt)
    tiebreak_messages = _build_messages(
        "logical_reasoning",
        prompt + "\n\n[Carefully re-check every constraint one at a time before giving your final answer.]",
    )

    async def _call(msgs: list) -> Optional[str]:
        if time.monotonic() >= deadline - 5:
            return None
        async with semaphore:
            r = await call_fireworks(config, model, msgs, category="logical_reasoning",
                                     deadline=deadline, max_tokens=max_tokens)
        if r.usage:
            token_tracker.record(r.usage, task_id=task_id, category="logical_reasoning", model=model)
        return r.content.strip() if r.success and r.content else None

    a1, a2 = await asyncio.gather(_call(messages), _call(messages))
    if a1 is None and a2 is None:
        return None
    if a1 is None:
        return a2
    if a2 is None:
        return a1

    c1 = strip_reasoning_blocks(a1).lower().strip()
    c2 = strip_reasoning_blocks(a2).lower().strip()
    if c1 == c2:
        logger.info("Task %r logical_reasoning: both calls agree", task_id)
        return a1

    logger.warning("Task %r logical_reasoning: disagreement (%r vs %r) — tie-breaking",
                   task_id, c1[:80], c2[:80])
    a3 = await _call(tiebreak_messages)
    if a3 is None:
        return a1
    c3 = strip_reasoning_blocks(a3).lower().strip()
    if c3 == c1:
        return a1
    if c3 == c2:
        return a2
    return a3  # tie-breaker is its own answer — use it


async def _process_task(task: dict, config, deadline: float, semaphore: asyncio.Semaphore) -> dict:
    task_id = task["task_id"]
    raw_prompt = task["prompt"].strip()

    # Early exit for empty prompts — no API call, no tokens wasted
    if not raw_prompt:
        logger.warning(
            "Task %r has an empty prompt — skipping API call and using placeholder answer",
            task_id,
        )
        return {
            "task_id": task_id,
            "answer": "No answer available: the task prompt was empty.",
        }

    prompt = _truncate_prompt(raw_prompt)
    category = classify_task(prompt, task_id)
    max_tokens = MAX_TOKENS_BY_CATEGORY.get(category, 512)

    # logical_reasoning: self-consistency before standard retry loop
    if category == "logical_reasoning":
        model = get_model_for_category(category)
        raw = await _logical_reasoning_with_consistency(
            task_id, prompt, config, model, max_tokens, deadline, semaphore
        )
        if raw:
            raw = _post_process(category, raw)
            is_valid, cleaned, _ = validate_and_finalize(task_id, category, raw, 0, prompt=prompt)
            return {"task_id": task_id, "answer": cleaned if is_valid else extract_last_resort(raw, category)}
        # all consistency calls failed — fall through to standard loop

    best_raw: Optional[str] = None
    retry_suffix = ""

    for attempt in range(3):
        if time.monotonic() >= deadline - 5:
            logger.warning("Task %r: deadline too close, using best available answer", task_id)
            break

        # Re-resolve model each attempt so warm-up failures mid-run are respected
        model = get_model_for_category(category)
        messages = _build_messages(category, prompt + retry_suffix)

        async with semaphore:
            result = await call_fireworks(
                config, model, messages, category=category,
                deadline=deadline, max_tokens=max_tokens,
            )

        if result.usage:
            token_tracker.record(result.usage, task_id=task_id, category=category, model=model)

        if not result.success:
            logger.warning("Task %r attempt %d failed on %s: %s", task_id, attempt, model, result.error)
            # If this model is dead (404 exhausted), mark it and re-route next attempt
            if "404" in (result.error or ""):
                mark_model_unavailable({model})
                logger.warning("Marked %s unavailable — next attempt will use fallback model", model)
            retry_suffix = f"\n\n[Previous attempt failed: {result.error}. Please try again.]"
            continue

        raw = _post_process(category, result.content)
        is_valid, cleaned, reason = validate_and_finalize(task_id, category, raw, attempt, prompt=prompt)

        if is_valid:
            short_retry = check_min_completeness(category, cleaned, prompt)
            if short_retry and attempt < 2:
                retry_suffix = f"\n\n[{short_retry}]"
                best_raw = cleaned
                continue
            return {"task_id": task_id, "answer": cleaned}

        best_raw = raw if raw else best_raw
        logger.warning("Task %r attempt %d invalid: %s", task_id, attempt, reason)
        if reason and "too long" in reason.lower():
            retry_suffix = "\n\n[Do not show your reasoning. Output only the final answer in the exact format specified.]"
        else:
            retry_suffix = f"\n\n[Your previous response was invalid: {reason}. Please correct it.]"

    # All attempts done — use best available with last-resort extraction if there was a reasoning leak
    if best_raw:
        fallback = extract_last_resort(best_raw, category)
    else:
        fallback = f"Unable to generate a confident answer for this task (task_id={task_id})."
    logger.warning("Task %r using fallback answer", task_id)
    return {"task_id": task_id, "answer": fallback}


async def _run(config, tasks: List[dict]) -> List[dict]:
    # Warm up Gemma models first — deadline starts AFTER this completes
    failed_warmup = await _warmup_gemma_models(config, config.allowed_models)
    if failed_warmup:
        mark_model_unavailable(failed_warmup)

    # Task-processing deadline starts now, after warm-up
    start = time.monotonic()
    deadline = start + DEADLINE_SECONDS
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    results_map: Dict[str, dict] = {}

    async def safe_process(task):
        try:
            res = await _process_task(task, config, deadline, semaphore)
        except Exception as e:
            logger.error("Unexpected error for task %r: %s\n%s",
                         task["task_id"], e, traceback.format_exc())
            res = {"task_id": task["task_id"], "answer": f"Processing error: {e}"}
        results_map[task["task_id"]] = res

    coros = [safe_process(t) for t in tasks]

    try:
        await asyncio.wait_for(
            asyncio.gather(*coros),
            timeout=DEADLINE_SECONDS - 10,
        )
    except asyncio.TimeoutError:
        logger.warning("Global deadline reached — filling in fallback answers for unprocessed tasks")

    # Ensure every task_id has an entry
    for task in tasks:
        if task["task_id"] not in results_map:
            results_map[task["task_id"]] = {
                "task_id": task["task_id"],
                "answer": "Processing did not complete within the time budget.",
            }

    elapsed = time.monotonic() - start
    token_tracker.log_summary()
    token_tracker.write_token_log(tasks_file=os.environ.get("TASKS_PATH", ""))
    logger.info("Completed %d tasks in %.1fs (excluding warm-up)", len(results_map), elapsed)

    return [results_map[t["task_id"]] for t in tasks]


def main():
    config = load_config()
    logger.info("Config loaded. Base URL: %s  Models: %s", config.base_url, config.allowed_models)

    init_router(config.allowed_models)

    tasks = read_tasks()
    results = []

    try:
        if tasks:
            results = asyncio.run(_run(config, tasks))
        else:
            logger.info("No tasks to process")
    except Exception as e:
        logger.error("Fatal error during processing: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            write_results(results)
        except Exception as e:
            logger.error("Failed to write results: %s", e)
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
