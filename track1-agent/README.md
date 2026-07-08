# Track 1 — General-Purpose AI Agent

Batch AI agent for the AMD Developer Hackathon. Reads `/input/tasks.json`, routes each task to the appropriate Fireworks model, and writes `/output/results.json`.

## Quick Start (local)

```bash
cp .env.example .env
# Fill in your FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS

export LOCAL_DEV=true
export TASKS_PATH=tests/sample_tasks/tasks_basic.json
pip install -r requirements.txt
python src/main.py
```

## Environment Variables (required at runtime)

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Bearer token for Fireworks API |
| `FIREWORKS_BASE_URL` | Base URL, e.g. `https://api.fireworks.ai/inference` |
| `ALLOWED_MODELS` | Comma-separated list of allowed model IDs |

`LOCAL_DEV=true` enables loading a local `.env` file (never set in production).

## Build & Push

```bash
# CRITICAL: always use --platform linux/amd64
REGISTRY=docker.io TEAM_NAME=your-team bash scripts/build_and_push.sh
```

## Run Tests

```bash
pip install pytest
pytest tests/
```

## Project Structure

```
src/
  main.py            # Orchestrator (asyncio, deadline-aware)
  config.py          # Env var loading & validation
  io_handler.py      # Read tasks.json / write results.json (atomic)
  classifier.py      # Rule-based task category classifier
  model_router.py    # Category → model tier → model ID
  fireworks_client.py # HTTP client with retries/timeouts
  validator.py       # Per-category answer validation & retry logic
  token_tracker.py   # Token usage logging
  logger_setup.py    # Logging config
  prompts/           # Per-category prompt builders
```

## Post-Launch-Day Tuning

1. Update `CATEGORY_TIER` in `model_router.py` with real model IDs.
2. Adjust `_SIZE_HINTS` in `model_router.py` if model names don't contain size hints.
3. Re-run `tasks_all_categories.json` to validate accuracy and re-tune tiers.
4. Rebuild and push the final image before the submission deadline.
