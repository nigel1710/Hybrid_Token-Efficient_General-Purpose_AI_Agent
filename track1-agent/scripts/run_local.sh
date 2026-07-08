#!/usr/bin/env bash
set -euo pipefail

# Copy .env.example to .env and fill in your real keys before running this script
export LOCAL_DEV=true
export TASKS_PATH="${TASKS_PATH:-tests/sample_tasks/tasks_basic.json}"

echo "Running pipeline locally against: $TASKS_PATH"
python src/main.py

echo ""
echo "=== Output ==="
cat /output/results.json 2>/dev/null || cat output/results.json 2>/dev/null || echo "(no output file found)"
