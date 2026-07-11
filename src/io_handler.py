import json
import logging
import os
import sys
from typing import List, Dict

logger = logging.getLogger(__name__)

TASKS_PATH = "input/tasks.json"
OUTPUT_PATH = "output/results.json"


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
            continue
        else:
            seen_ids[tid] = i
        if not item["prompt"] or not str(item["prompt"]).strip():
            logger.warning("Task %r has empty prompt", tid)
        tasks.append({"task_id": str(tid), "prompt": str(item.get("prompt", ""))})

    return tasks


def write_results(results: List[Dict]):
    abs_path = os.path.abspath(OUTPUT_PATH)
    output_dir = os.path.dirname(abs_path)
    os.makedirs(output_dir, exist_ok=True)

    output = [{"task_id": r["task_id"], "answer": r["answer"]} for r in results]
    json_str = json.dumps(output, ensure_ascii=False, indent=2)

    # Self-check string validity before write
    json.loads(json_str)

    tmp_path = abs_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    
    # os.replace is correct cross-platform call for Windows and Linux
    os.replace(tmp_path, abs_path)
    
    # Read-back verification
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            read_back_data = json.load(f)
        if not isinstance(read_back_data, list):
            logger.error("Read-back verification FAILED: Content at %s is not a list", abs_path)
        elif len(read_back_data) != len(output):
            logger.error(
                "Read-back verification FAILED: Code intended to write %d entries, "
                "but on-disk file %s actually contains %d entries",
                len(output), abs_path, len(read_back_data)
            )
        else:
            logger.info("Read-back verification PASSED: Confirmed %d entries exist at %s", len(read_back_data), abs_path)
    except Exception as e:
        logger.error("Read-back verification FAILED to read from %s: %s", abs_path, e)

    logger.info("Wrote %d results to %s", len(output), abs_path)
