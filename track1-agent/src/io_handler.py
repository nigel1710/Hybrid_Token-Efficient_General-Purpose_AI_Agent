import json
import logging
import os
import sys
from typing import List, Dict

logger = logging.getLogger(__name__)

TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
OUTPUT_PATH = "/output/results.json"


def read_tasks() -> List[Dict]:
    path = TASKS_PATH
    if not os.path.exists(path):
        logger.error("tasks.json not found at %s", path)
        sys.exit(1)

    raw = open(path, "r", encoding="utf-8").read().strip()
    if not raw:
        logger.error("tasks.json is empty at %s", path)
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("tasks.json is not valid JSON: %s", e)
        sys.exit(1)

    if not isinstance(data, list):
        logger.error("tasks.json must be a JSON array, got %s", type(data).__name__)
        sys.exit(1)

    if len(data) == 0:
        logger.info("tasks.json is an empty array — nothing to process")
        return []

    tasks = []
    seen_ids = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error("Task at index %d is not an object", i)
            sys.exit(1)
        if "task_id" not in item or "prompt" not in item:
            logger.error("Task at index %d missing 'task_id' or 'prompt'", i)
            sys.exit(1)
        tid = item["task_id"]
        if tid in seen_ids:
            logger.warning("Duplicate task_id %r at index %d (first seen at %d)", tid, i, seen_ids[tid])
        else:
            seen_ids[tid] = i
        if not item["prompt"] or not str(item["prompt"]).strip():
            logger.warning("Task %r has empty prompt", tid)
        tasks.append({"task_id": str(tid), "prompt": str(item.get("prompt", ""))})

    return tasks


def write_results(results: List[Dict]):
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    output = [{"task_id": r["task_id"], "answer": r["answer"]} for r in results]
    json_str = json.dumps(output, ensure_ascii=False, indent=2)

    # Self-check
    json.loads(json_str)

    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    os.replace(tmp_path, OUTPUT_PATH)
    logger.info("Wrote %d results to %s", len(output), OUTPUT_PATH)
