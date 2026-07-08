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

## Model Routing (real list now wired in)

The five Track 1 models are now explicitly mapped in `model_router.py`:

| Model | Tier | Notes |
|---|---|---|
| `gemma-4-26b-a4b-it` | CHEAP | MoE, small active params — fastest/cheapest |
| `gemma-4-31b-it-nvfp4` | CHEAP | Quantized 31b — near-CHEAP cost, more accurate than a4b |
| `gemma-4-31b-it` | MID | Full 31b general model |
| `kimi-k2p7-code` | override | Fixed routing for `code_debugging` + `code_generation` |
| `minimax-m3` | LARGE | `math_reasoning` + `logical_reasoning` |

Set `ALLOWED_MODELS` in your `.env` to the full comma-separated list:
```
ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
```

## Pre-Submission Validation Checklist

**Run the full local test batch against real Fireworks calls before final submission:**

```bash
export LOCAL_DEV=true
export TASKS_PATH=tests/sample_tasks/tasks_all_categories.json
python src/main.py
```

For each category in the output, verify the answer quality:
- If `sentiment_classification` or `ner` underperform on `gemma-4-26b-a4b-it`, bump them to `MID`
- If `factual_knowledge` or `summarisation` underperform on `gemma-4-31b-it`, bump to `LARGE`
- If `math_reasoning` or `logical_reasoning` underperform on `minimax-m3`, there is no higher tier — consider adding chain-of-thought prompting instead
- If code categories underperform on `kimi-k2p7-code`, check the prompt templates in `src/prompts/code_debugging.py` and `src/prompts/code_generation.py` first

Retune `_MODEL_TIERS` and `CATEGORY_TIER` in `model_router.py` based on those results, then rebuild and push the final image.
