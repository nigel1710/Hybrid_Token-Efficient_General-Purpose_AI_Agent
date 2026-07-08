# Track 1 — General-Purpose AI Agent: Full Implementation Guide

**Purpose of this document:** This is a complete, step-by-step build specification for implementing the AMD Developer Hackathon Act II, Track 1 submission. It is written to be handed directly to an AI coding assistant (e.g., Claude Code) to implement the entire project from scratch, file by file, with all edge cases considered.

**Team context:** Two engineers, basic-to-intermediate level in AI/ML engineering. Prioritize **correctness, reliability, and simplicity** over cleverness. Every design decision below optimizes first for "does not crash / does not produce malformed output / passes the accuracy gate" and second for "uses as few tokens as possible."

---

## 0. High-Level Requirements Recap (Ground Truth)

- Input: `/input/tasks.json` — array of `{ "task_id": string, "prompt": string }`
- Output: `/output/results.json` — array of `{ "task_id": string, "answer": string }`, written before the process exits
- Must read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` (comma-separated) from environment variables at runtime — never hardcoded, never bundled in a `.env` inside the shipped image
- ALL inference calls must go through `FIREWORKS_BASE_URL`. Local models score zero tokens and are not counted — do not use local inference for actual answers.
- Only models listed in `ALLOWED_MODELS` may be called
- Exit code `0` on success, non-zero on failure
- Max total runtime: 10 minutes
- Max container startup time: 60 seconds
- Max response time per request: 30 seconds
- Image must be `linux/amd64`, compressed size ≤ 10GB
- No hardcoded/cached answers — evaluation uses unseen prompt variants
- All responses must be in English
- 8 task categories: factual knowledge, mathematical reasoning, sentiment classification, text summarisation, named entity recognition, code debugging, logical/deductive reasoning, code generation
- Scoring: (1) LLM-Judge accuracy gate (pass/fail threshold) → (2) among passers, rank ascending by total tokens used

---

## 1. Project Structure

Instruct the coding assistant to create this exact directory layout:

```
track1-agent/
├── Dockerfile
├── requirements.txt
├── .env.example
├── .dockerignore
├── README.md
├── src/
│   ├── main.py                  # Entry point / orchestrator
│   ├── config.py                 # Env var loading & validation
│   ├── io_handler.py              # Reading tasks.json, writing results.json
│   ├── classifier.py               # Rule-based category classification
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── factual.py
│   │   ├── math_reasoning.py
│   │   ├── sentiment.py
│   │   ├── summarisation.py
│   │   ├── ner.py
│   │   ├── code_debugging.py
│   │   ├── logical_reasoning.py
│   │   └── code_generation.py
│   ├── model_router.py             # Maps category -> model tier -> actual model ID
│   ├── fireworks_client.py          # HTTP client wrapper with retries/timeouts
│   ├── validator.py               # Per-answer and per-file validation
│   ├── token_tracker.py            # Local token usage estimator/logger (for dev insight only)
│   └── logger_setup.py             # Structured logging config
├── tests/
│   ├── sample_tasks/
│   │   ├── tasks_basic.json
│   │   ├── tasks_edge_cases.json
│   │   └── tasks_all_categories.json
│   ├── test_classifier.py
│   ├── test_validator.py
│   └── test_end_to_end.py
└── scripts/
    ├── run_local.sh
    └── build_and_push.sh
```

Instruct the assistant to create every file listed, even if some start out minimal, so the structure is complete and importable.

---

## 2. Environment & Configuration (`src/config.py`)

### Requirements
- Load `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` strictly from `os.environ`.
- Support a **local-only** `.env` file (via `python-dotenv`) that is loaded ONLY if a `LOCAL_DEV` environment variable is set to `"true"`. This must never be loaded in the production container path — guard it explicitly so it cannot silently pick up a bundled `.env` at evaluation time.
- `.env` must be listed in `.dockerignore` so it can never accidentally ship inside the image even if present in the build context.
- Validate on startup:
  - `FIREWORKS_API_KEY` is non-empty
  - `FIREWORKS_BASE_URL` is a valid URL (has scheme + host)
  - `ALLOWED_MODELS` is non-empty and splits into at least one model ID after stripping whitespace and removing empty entries
- If any required variable is missing or invalid, log a clear error to stderr and exit immediately with a non-zero exit code — **do not attempt any Fireworks calls** if config is broken, since that would waste the runtime budget and guarantee failure anyway.
- Expose a single `Config` object/dataclass that the rest of the app imports, so there is exactly one source of truth for these values.

### Edge cases to handle here
- `ALLOWED_MODELS` has trailing/leading commas or extra whitespace (`"model-a, model-b,"`) — parse defensively.
- `FIREWORKS_BASE_URL` has a trailing slash or missing trailing slash — normalize so URL joining doesn't produce `//v1/chat/completions` or similar.
- Environment variables present but empty strings (`FIREWORKS_API_KEY=""`) — treat as missing, not as valid.

---

## 3. Input/Output Handling (`src/io_handler.py`)

### Reading input
- Read from `/input/tasks.json`. Path should be configurable via a constant at the top of the file (not hardcoded deep in logic) so it's easy to change for local testing (e.g., point to `tests/sample_tasks/*.json`).
- Handle and log clearly, then exit non-zero, for:
  - File does not exist at the expected path
  - File exists but is empty
  - File is not valid JSON (catch `json.JSONDecodeError`)
  - JSON is valid but not a list
  - JSON is a list but contains items missing `task_id` or `prompt`
  - Duplicate `task_id` values in the input (log a warning, keep both, use array index as a tiebreaker internally if needed, but preserve both task_ids in output)
  - Empty `prompt` string for a given task (still must produce SOME answer for that task_id in output — see fallback logic in Validator section)
  - `tasks.json` is an empty array `[]` — this is valid; produce an empty `results.json` array and exit 0 successfully.

### Writing output
- Always write to `/output/results.json`, creating the `/output` directory if it doesn't exist (defensive — should exist per harness, but don't assume).
- Write using an atomic pattern: write to a temp file (`results.json.tmp`) in the same directory, then rename to `results.json`. This avoids ever leaving a half-written/corrupt file on disk if the process is killed mid-write.
- Before finalizing, run the file back through `json.loads` on the string you're about to write, as a last self-check, to guarantee it is valid JSON syntactically.
- Ensure every `task_id` present in the input appears exactly once in the output. This is critical — the spec says missing entries could be scored as failures.
- Output field order/casing: use exactly `task_id` and `answer` as keys, matching the spec precisely.

---

## 4. Task Classification (`src/classifier.py`)

### Approach: Rule-based heuristics first, no LLM call for classification (saves tokens)

Implement a function `classify_task(prompt: str) -> str` that returns one of:
`"factual_knowledge"`, `"math_reasoning"`, `"sentiment_classification"`, `"summarisation"`, `"ner"`, `"code_debugging"`, `"logical_reasoning"`, `"code_generation"`

### Heuristic rules (apply in this priority order, first match wins)

1. **Code debugging**: prompt contains a code block (triple backticks, or indented code, or common language keywords like `def `, `function`, `class `, `import`, `#include`) AND contains bug/error/fix-related words (`bug`, `error`, `fix`, `debug`, `not working`, `incorrect output`, `exception`, `traceback`).
2. **Code generation**: prompt contains phrases like `write a function`, `implement`, `write code`, `create a program`, `write a script` AND does NOT already contain an existing code block with an error to fix (to disambiguate from debugging).
3. **Named entity recognition**: prompt mentions `extract entities`, `named entities`, `identify all`, or explicitly lists entity types (`person`, `organization`, `location`, `date`) alongside an extraction verb.
4. **Text summarisation**: prompt contains `summarise`, `summarize`, `condense`, `tl;dr`, `shorten`, or gives an explicit length constraint (`in one sentence`, `in 3 bullet points`, `under 50 words`) combined with a block of source text to compress.
5. **Sentiment classification**: prompt contains `sentiment`, `classify the tone`, `positive or negative`, `how does the reviewer feel`.
6. **Mathematical reasoning**: prompt contains numbers plus operators/keywords (`%`, `calculate`, `how many`, `total`, `average`, `if x and y`, word-problem structure) AND is not primarily a code task.
7. **Logical/deductive reasoning**: prompt contains puzzle-style constraint language (`exactly one of`, `must be true`, `if and only if`, `either...or`, multiple named entities with relational constraints, classic puzzle framing like "A, B, and C each...").
8. **Default / fallback: Factual knowledge** — if nothing else matches, treat as a factual/explanatory question.

### Implementation notes for the coding assistant
- Implement this as a series of independent boolean-returning helper functions (`_looks_like_code_debugging(text)`, etc.) composed in the main `classify_task` function, so each rule is independently testable and tunable.
- Make matching case-insensitive.
- Log the classification decision for every task (task_id → chosen category) at INFO level, so it's easy to audit misclassifications during testing.
- Add a `confidence` concept internally even if not exposed in output: if a prompt matches multiple category signals strongly (e.g., both "summarise" and "extract entities" appear), fall back to whichever rule is highest priority per the ordered list above, but log a WARNING noting the ambiguity for later manual review.
- **Do not** call an LLM for classification in the first version. Only add an LLM-based fallback classifier (a single cheap-model call used ONLY when zero heuristics match) as a stretch goal, and gate it behind a config flag so it can be disabled to save tokens.

### Edge cases
- Extremely short prompts (e.g., `"2+2"`) — should still hit the math heuristic via digit/operator detection.
- Prompts mixing categories (e.g., "Summarise this text and extract the named entities") — pick the FIRST instruction/verb that appears in the text as the primary category, since output format only supports one `answer` field per task; log this ambiguity clearly.
- Non-English prompts — classify as best-effort using the same rules (language of the prompt doesn't block classification), but note that your final answer must still be in English per the rules.

---

## 5. Model Routing / Tiering (`src/model_router.py`)

### Purpose
Map each category to a **model tier** (cheap / mid / large), then resolve that tier to an actual model ID from `ALLOWED_MODELS` at runtime — never hardcode specific model ID strings, since `ALLOWED_MODELS` is only published on launch day and may change.

### Design
- On startup, after reading `ALLOWED_MODELS`, build an ordered list of models from cheapest/smallest to largest/most capable. Since exact model identities aren't known yet, implement this ordering using a **configurable heuristic**: sort by common naming conventions (e.g., parameter count substrings like `8b`, `70b`, `405b` if present in model IDs) with a manual override list that the team can edit once real model IDs are published on launch day.
- Expose three logical tiers: `CHEAP`, `MID`, `LARGE`. If `ALLOWED_MODELS` has fewer than 3 entries, gracefully collapse tiers (e.g., 2 models → CHEAP=model[0], MID=LARGE=model[1]; 1 model → all tiers point to the same model).
- Category → tier mapping (initial default, tune after testing):

| Category | Default Tier |
|---|---|
| sentiment_classification | CHEAP |
| ner | CHEAP |
| factual_knowledge | CHEAP or MID (test both) |
| summarisation | MID |
| code_debugging | MID |
| math_reasoning | LARGE |
| logical_reasoning | LARGE |
| code_generation | LARGE or MID (test both — many code-gen tasks are simple) |

- Make this mapping a plain, clearly-commented dictionary/config at the top of `model_router.py` so the team can quickly re-tune it after empirical testing without touching orchestration logic.
- Provide a single function `get_model_for_category(category: str) -> str` that the orchestrator calls.

### Edge cases
- `ALLOWED_MODELS` published with unexpected/unfamiliar naming (no clear size hints in the string) — the sort heuristic must not crash; fall back to using the raw order given in `ALLOWED_MODELS` (treat first as CHEAP, middle as MID, last as LARGE) if size-parsing fails.
- A configured tier model ID is not actually present in `ALLOWED_MODELS` at runtime (e.g., stale manual override list) — validate at startup that every model the router might select is actually in `ALLOWED_MODELS`; if not, fall back to any valid model from `ALLOWED_MODELS` and log a loud warning rather than crashing.

---

## 6. Fireworks API Client (`src/fireworks_client.py`)

### Requirements
- Wraps calls to `FIREWORKS_BASE_URL` (OpenAI-compatible chat completions style, standard for Fireworks AI — confirm exact endpoint path against Fireworks docs once available, default assumption: `POST {FIREWORKS_BASE_URL}/chat/completions` or `/v1/chat/completions` — make this path a single named constant so it's a one-line change if the harness specifies something different).
- Use `FIREWORKS_API_KEY` as a Bearer token in the `Authorization` header.
- Support:
  - Configurable per-request timeout, default 25 seconds (leaving margin under the 30-second hard limit for network/processing overhead).
  - Retry logic: on timeout, 429 (rate limit), or 5xx response, retry up to 2 additional times with short exponential backoff (e.g., 1s, 2s). Do NOT retry on 4xx errors other than 429 (e.g., a 400 bad request means the payload itself is wrong — retrying won't help, log and fall through to fallback handling instead).
  - Respect the overall 10-minute runtime budget: pass in a "deadline" concept from the orchestrator so the client can refuse to start a new call (and instead trigger fallback handling) if there isn't enough time left.
- Log, for every call: category, model used, prompt token estimate (if available from response usage field), completion token estimate, latency, and success/failure status. This is for the team's own tuning — NOT sent anywhere externally.
- Never log the full raw `FIREWORKS_API_KEY` value, not even in debug logs.
- Return a structured result object: `{ success: bool, content: str | None, error: str | None, usage: {...} | None }` so the orchestrator can branch cleanly.

### Critical correctness requirement
- **Every single model call in this file must be constructed using `Config.base_url` and a model ID resolved via `model_router`.** There must be no code path anywhere in the project that calls any other base URL or any model string not sourced from `ALLOWED_MODELS`. Instruct the coding assistant to grep the finished codebase for any hardcoded URL or model string literal before finalizing, as a submission-ending mistake to avoid.

### Edge cases
- Fireworks returns a response with unexpected schema (missing `choices[0].message.content`) — handle gracefully, treat as a failed call, do not crash with an unhandled `KeyError`.
- Fireworks returns an empty string as the answer — treat as invalid, trigger retry/fallback logic (see Validator).
- Network completely unreachable (DNS failure, connection refused) — catch the specific exception type, log clearly, treat as failed call, allow orchestrator's fallback path to proceed rather than the whole container crashing.
- Response takes longer than the timeout — must be a hard client-side timeout (not just "hope the server responds"), so one hung request cannot consume the whole 10-minute budget.

---

## 7. Prompt Templates (`src/prompts/*.py`)

Each file in `src/prompts/` should expose a single function like `build_prompt(task_prompt: str) -> dict` (or a `(system, user)` tuple) tailored to that category. General principles for every template:

- Be explicit about the exact expected output format (plain text, single number, JSON, code block, etc.) so the model doesn't produce extra prose that costs tokens and risks failing strict parsing.
- Instruct the model NOT to repeat the original question back, NOT to add disclaimers/hedging language, and NOT to wrap answers in unnecessary markdown unless the category needs it (e.g., code).
- Keep system prompts short — a few sentences, not paragraphs — since system prompt tokens are paid on every single call.

### Per-category template specs

**`factual.py`**
- System: "Answer the question directly and concisely. Do not repeat the question. No preamble."
- No few-shot example needed; these are typically short.

**`math_reasoning.py`**
- System: "Solve the problem. You may reason internally, but output ONLY the final answer clearly on the last line, prefixed with 'Answer: '."
- Note for coding assistant: after receiving the raw model output, parse out the final answer using a small post-processing step (look for text following `Answer:`); if that prefix is missing, fall back to using the full trimmed response as the answer rather than failing the task.

**`sentiment.py`**
- System: "Classify the sentiment of the text as exactly one of: positive, negative, neutral. Then give a one-sentence justification. Format: 'Sentiment: <label>. Justification: <reason>'."
- Post-processing: validate the label is exactly one of the 3 allowed values (case-insensitive); if the model returns something else (e.g., "mixed"), keep the raw text as the answer but log a warning — do not silently force it into a wrong bucket, since intent-based judging should still credit a reasonable answer.

**`summarisation.py`**
- System: "Summarise the following text according to the exact length/format instruction given. Do not exceed the specified constraint."
- Include the original length constraint verbatim, extracted from the task prompt, restated back to the model.
- Post-processing: if the task specified e.g. "one sentence" and the model returns multiple sentences, this is a candidate for one retry with a stricter reminder appended.

**`ner.py`**
- System: "Extract all named entities from the text. Return ONLY valid JSON with this exact structure: {\"person\": [...], \"organization\": [...], \"location\": [...], \"date\": [...]}. Use empty arrays for categories with no matches. Do not include any text outside the JSON object."
- Post-processing: attempt `json.loads` on the model's output. If it fails, try stripping markdown code fences (```json ... ```) and retry parsing. If still invalid, trigger the retry-with-correction flow (see Validator section) before falling back.

**`code_debugging.py`**
- System: "The following code has a bug. Identify it in one short sentence, then provide the corrected code in a single code block. Preserve the original structure and style; change only what's necessary to fix the bug."
- Include the original code snippet from the task prompt verbatim inside the user message.

**`logical_reasoning.py`**
- System: "Solve this logic puzzle. Reason through all constraints internally, then output only the final answer(s) that satisfy every condition, clearly and concisely."
- These are allowed more reasoning tokens since correctness is paramount and typically model has to reason step-by-step to get these right — this is the one category where verbosity is an acceptable tradeoff for accuracy; still ask for the final answer to be clearly delimited.

**`code_generation.py`**
- System: "Write a correct, well-structured function based on the specification. Return ONLY the code in a single code block, no explanation text before or after."
- Post-processing: strip any leading/trailing prose outside the code block if the model disobeys, so the stored answer is clean code.

---

## 8. Validation & Retry Layer (`src/validator.py`)

This is the most important reliability component. Implement a function `validate_and_finalize(task_id, category, raw_answer, attempt_number) -> (is_valid: bool, cleaned_answer: str | None, retry_reason: str | None)`.

### General validation rules (apply to all categories)
- Reject if `raw_answer` is `None` or an empty/whitespace-only string.
- Reject if the raw answer looks like an API error message leaking through (e.g., contains substrings like "I cannot", "As an AI language model I don't have access" combined with an otherwise empty substantive answer) — this is a softer heuristic check, log but don't over-block genuine short valid answers.
- Trim excessive whitespace/newlines from the final stored answer.

### Category-specific validation
- **NER**: must parse as valid JSON with the four expected keys present (missing keys should be auto-filled with empty arrays rather than failing validation outright, since partial-but-valid JSON is recoverable).
- **Sentiment**: should contain a recognizable label; if not, still valid but logged as a soft-fail for review.
- **Code generation / debugging**: check that a code block delimiter or clearly code-like content exists (very basic heuristic: contains `def `, `{`, `;`, `function`, etc.) — this is a soft check, not a hard reject, since some correct answers might not use triple backticks.
- **Math reasoning**: check the answer isn't just a restatement of the question with no computed value; a soft heuristic (contains at least one digit somewhere, unless the expected answer is inherently non-numeric like "impossible").

### Retry policy
- Maximum **2 retries per task** (3 total attempts), to protect the overall 10-minute runtime and token budget.
- On retry, append a short corrective instruction to the prompt based on `retry_reason` (e.g., "Your previous response was not valid JSON. Return ONLY valid JSON matching the schema, nothing else.").
- If all attempts fail validation, use a **fallback answer**: store the best available raw response from the last attempt, even if imperfect, rather than an empty string — a partial answer has a chance to pass the LLM-Judge's intent check; an empty string guarantees zero credit.
- If ALL attempts return `None`/error (e.g., total API failure, not just a bad format), the fallback answer should be a short, honest, best-effort text (e.g., a plain-English attempt at answering constructed from whatever partial info is available) rather than leaving that `task_id` out of the output entirely — never omit a task_id from the final output file.

### Edge cases
- A task's `prompt` field is empty or whitespace-only in the input — still must produce a non-empty `answer` in the output; classify as `factual_knowledge` by default and have the model/fallback generate a generic clarifying-style answer rather than crashing on an empty prompt being sent to the model.
- Extremely long prompts near a model's context limit — truncate defensively with a clear log warning rather than letting the API call fail outright; prioritize keeping the end of a prompt (often contains the actual instruction) if truncation is required, while preserving the very beginning (often contains framing/context).

---

## 9. Orchestrator (`src/main.py`)

### Responsibilities, in order
1. Load and validate config (`config.py`). Exit immediately on failure.
2. Read and validate `tasks.json` (`io_handler.py`). Exit immediately on failure (except the empty-array case, which is valid).
3. Record a hard deadline: `start_time + 9.5 minutes` (leave a 30-second safety margin under the 10-minute limit for final file writes and cleanup).
4. For each task:
   a. Classify category.
   b. Resolve model via `model_router`.
   c. Build category-specific prompt.
   d. Call Fireworks client.
   e. Validate response; retry as needed (respecting the global deadline — if the deadline is close, skip further retries and go straight to fallback).
   f. Record `{task_id, answer}`.
5. **Concurrency**: process tasks concurrently (e.g., `asyncio` with a bounded semaphore, suggested concurrency limit of 5–10 simultaneous requests) rather than strictly sequentially, to make good use of the 10-minute window when there are many tasks. Ensure thread/async-safety when aggregating results (use a simple list with index-based insertion or a dict keyed by task_id, not a shared mutable list appended from multiple coroutines without synchronization).
6. If the global deadline is reached before all tasks are processed, immediately stop launching new calls, fill in a fallback answer (e.g., short apologetic-but-substantive best-effort text, or the raw unvalidated response if one was in flight) for any remaining unprocessed `task_id`s, and proceed to writing output — **never let the process run past its budget and get killed by the harness with zero output written**.
7. Write `/output/results.json` via `io_handler`.
8. Log a final summary (tasks processed, retries triggered, estimated total tokens used, total runtime) for the team's own review.
9. Exit code `0`.

### Top-level error handling
- Wrap the entire task-processing loop in a try/except that guarantees `/output/results.json` is still written with whatever partial results exist, even if an unexpected exception occurs partway through — a partial valid file scores something; a crash with no file scores zero for everything.
- Use a final `try/finally` or equivalent pattern so the write-output step always executes on the way out, whether processing succeeded, partially failed, or hit the deadline.
- Any uncaught exception anywhere in the pipeline should be caught at the top level, logged with a full traceback (to stderr/logs, not into the answers themselves), and result in exit code `1` **only if no output file could be produced at all**; if a partial/best-effort output file was successfully written despite the error, prefer exiting `0` since the file itself is what's graded.

---

## 10. Logging (`src/logger_setup.py`)

- Use Python's standard `logging` module, structured with timestamps, log level, and module name.
- Log to stdout/stderr (not to a file inside `/output`, since that directory is reserved for `results.json`).
- INFO level: startup config summary (without secrets), per-task classification and model selection, per-task success/failure, final summary stats.
- WARNING level: retries triggered, fallback answers used, ambiguous classifications, deadline-triggered early termination.
- ERROR level: unhandled exceptions, config validation failures, total API unreachability.
- Never log the raw `FIREWORKS_API_KEY`.

---

## 11. Dockerfile

### Requirements
- Base image: a slim Python image (e.g., `python:3.11-slim`) — no ML frameworks, no model weights, since all inference is remote via Fireworks.
- Copy only `requirements.txt` first and install dependencies, then copy source code — standard layer-caching best practice to keep rebuilds fast.
- Set `WORKDIR` appropriately (e.g., `/app`).
- Entrypoint should directly run `python src/main.py` — no shell wrapper scripts that add startup latency, since the container must be ready within 60 seconds.
- Do NOT copy any `.env` file into the image — enforce this via `.dockerignore`, and instruct the coding assistant to double check the final image with `docker run ... ls -la /app` to confirm no `.env` is present, as a manual verification step.
- Ensure the container's default command exits after processing (no long-running server, no `while True` loop) — this is a batch job, not a service.
- **Explicitly build with `--platform linux/amd64`** — this must be documented prominently in `scripts/build_and_push.sh` and the README, since building on Apple Silicon without this flag silently produces an incompatible image that scores zero.

### `.dockerignore` should include at minimum
```
.env
.env.*
__pycache__/
*.pyc
.git/
.gitignore
tests/
README.md
*.md
.venv/
venv/
```
(Keep `requirements.txt` and `src/` out of this ignore list, obviously — only ignore dev/test artifacts and secrets.)

---

## 12. `requirements.txt`

Keep this minimal and pinned to specific versions once decided, to avoid the image silently changing behavior between builds. Expected contents (finalize exact versions when implementing):
```
httpx
python-dotenv
pydantic
```
(`httpx` for async HTTP calls to Fireworks; `python-dotenv` for local-dev-only `.env` loading; `pydantic` optional but recommended for strict input/output schema validation of `tasks.json`/`results.json`.)

Avoid adding heavyweight packages (no `torch`, `transformers`, `numpy` unless genuinely needed) — every unnecessary dependency adds image size and startup time.

---

## 13. Local Testing Plan (`tests/` and `scripts/run_local.sh`)

### Sample test files to create
- `tests/sample_tasks/tasks_basic.json`: 1 example per category (8 total), clearly and unambiguously worded, to sanity-check the happy path end-to-end.
- `tests/sample_tasks/tasks_edge_cases.json`: include —
  - an empty-string prompt
  - a prompt mixing two category signals (e.g., "summarise and extract entities from this text")
  - a very long prompt (near context-limit length)
  - a math prompt with no clean numeric answer possible (e.g., a trick question)
  - a code debugging prompt with subtly wrong code
  - a logic puzzle with multiple constraints
  - duplicate `task_id` values across two entries
- `tests/sample_tasks/tasks_all_categories.json`: a larger, more realistic batch (e.g., 3–5 per category = 24–40 tasks) to test concurrency, runtime budget, and token usage at a more representative scale.

### `scripts/run_local.sh`
- Should set `LOCAL_DEV=true`, load a local `.env` (team's own Fireworks trial/dev key if available, or a mock/staging endpoint), mount a chosen sample tasks file to `/input/tasks.json` equivalent path (or point `TASKS_PATH` env var at it, if the coding assistant makes that path configurable for local runs), run the container or run `python src/main.py` directly, and print the resulting `/output/results.json` content plus a summary of runtime and estimated tokens.

### Unit tests
- `test_classifier.py`: assert each sample prompt in `tasks_basic.json` classifies into its intended category; include a few adversarial phrasing variants per category to check heuristic robustness (this is exactly where "unseen prompt variants" risk shows up, so invest real effort here).
- `test_validator.py`: test that malformed JSON strings for NER are either repaired or correctly flagged for retry; test that empty answers are rejected; test that the final fallback logic never returns `None`/empty for the stored answer.
- `test_end_to_end.py`: run the full pipeline against `tasks_edge_cases.json` with a mocked Fireworks client (no real API calls in CI) that returns deliberately malformed/slow/erroring responses, and assert that `results.json` is still fully valid and contains every `task_id` no matter what the mock does.

---

## 14. Build & Push (`scripts/build_and_push.sh`)

Document these exact steps for the team:
1. `docker buildx create --use` (if buildx isn't already set up as the active builder)
2. `docker buildx build --platform linux/amd64 --tag <registry>/<team-name>/track1-agent:latest --push .`
3. After push, pull-test on a clean machine/VM if possible: `docker pull <registry>/<team-name>/track1-agent:latest` and run it against a local `tasks.json` with real (or the published) `FIREWORKS_*` env vars to confirm it behaves identically to local dev.
4. Confirm image size with `docker images` and ensure it is comfortably under the 10GB cap (expect well under 1GB for this design).
5. Confirm the registry repository is set to **public** visibility before the submission deadline — a private image will fail to pull and score zero.

---

## 15. Full Edge Case Checklist (consolidated)

Instruct the coding assistant to explicitly write tests or defensive code for every item below:

**Input handling**
- Missing `/input/tasks.json`
- Empty file / empty array
- Malformed JSON
- Missing `task_id` or `prompt` fields on some entries
- Duplicate `task_id`s
- Empty-string `prompt`
- Extremely long `prompt` (context-limit risk)
- Non-English prompt text
- Prompt mixing multiple category signals

**Config / environment**
- Missing or empty `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS`
- Malformed `ALLOWED_MODELS` (extra commas/whitespace)
- `FIREWORKS_BASE_URL` with/without trailing slash
- Accidentally bundled `.env` inside the image

**Model/API behavior**
- Fireworks timeout
- Fireworks 429 rate limit
- Fireworks 5xx error
- Fireworks 4xx error (bad request, not rate-limit)
- Malformed/unexpected response schema
- Empty completion string returned
- Network unreachable / DNS failure
- A configured tier's model ID not actually present in `ALLOWED_MODELS`

**Processing logic**
- Runtime approaching the 10-minute cap mid-batch
- One request hanging near the 30-second cap
- NER output not valid JSON
- Sentiment output using a label outside {positive, negative, neutral}
- Summarisation ignoring the requested length constraint
- Code answer wrapped in unexpected formatting or missing entirely
- All retries exhausted for a task
- Concurrent task processing producing race conditions in result aggregation

**Output handling**
- Partial results if the process is interrupted
- Ensuring every input `task_id` appears exactly once in output
- Final JSON self-validation before writing
- Atomic file write (temp file + rename)

**Docker / infra**
- Building on Apple Silicon without `--platform linux/amd64`
- Image exceeding size limits due to unnecessary dependencies
- Container startup exceeding 60 seconds
- Registry image left private

---

## 16. Suggested Build Order (for the coding assistant to follow sequentially)

1. Scaffold the full directory/file structure (empty stubs where needed).
2. Implement `config.py` + tests for env var validation.
3. Implement `io_handler.py` (read/write + validation) + tests using sample files.
4. Implement `classifier.py` + `test_classifier.py` against `tasks_basic.json` and edge cases.
5. Implement `model_router.py` with a mocked `ALLOWED_MODELS` string for testing tier resolution logic.
6. Implement `fireworks_client.py` with retry/timeout logic, initially tested against a mock server or mocked HTTP responses (do not burn real tokens during unit tests).
7. Implement all 8 prompt template modules.
8. Implement `validator.py` + `test_validator.py`, including the fallback-answer guarantee.
9. Wire everything together in `main.py`, including concurrency, deadline handling, and the guaranteed-output-write-on-exit pattern.
10. Run full local end-to-end tests with a real (dev/trial) Fireworks key against `tasks_all_categories.json`; measure actual token usage and tune the model-tiering table in `model_router.py` based on real accuracy/cost tradeoffs observed.
11. Write the Dockerfile, `.dockerignore`, `requirements.txt`.
12. Build locally with `--platform linux/amd64`, run the container end-to-end exactly as the harness would (mounting a test `tasks.json`, real env vars), confirm output.
13. Push to a public registry, pull-test from a clean environment.
14. Final review pass: grep the codebase for any hardcoded model ID, base URL, or leftover `.env`/secrets before final submission.

---

## 17. Post-Launch-Day Action Items (once `ALLOWED_MODELS` is published)

- Update the manual override list in `model_router.py` with the real model IDs and correct tier assignments based on their actual known capabilities/sizes.
- Re-run the full `tasks_all_categories.json` local test batch against the real models to confirm accuracy and re-tune tiering if a "cheap" model is failing certain categories.
- Re-confirm `FIREWORKS_BASE_URL` path structure matches the real Fireworks endpoint (adjust the constant in `fireworks_client.py` if needed).
- Re-build and re-push the final image after this tuning pass, well before the submission deadline, accounting for the 10-submissions-per-hour rate limit if multiple pushes/tests are needed.

---

*End of implementation guide. Feed this document to Claude Code as the full specification for building the project from scratch.*
