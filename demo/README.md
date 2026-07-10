# Live Demo — Streamlit Frontend

Interactive demo for the Track 1 General-Purpose AI Agent.
Runs the real classification, routing, and validation pipeline with a visual UI for judges.

**This is a demo layer only — it is not part of the Docker submission image.**

## Setup

```bash
pip install -r demo/requirements.txt
```

## Run

From the project root:

```bash
streamlit run demo/app.py
```

Then open http://localhost:8501 in your browser.

## What it does

- Upload or paste a `tasks.json` file (or load the built-in sample batch)
- Enter your Fireworks API credentials in the config panel
- Click **Run Agent** — each task is processed live with per-task status updates
- Switch to **Results Dashboard** for token breakdowns, model routing charts, and a downloadable `results.json`
- Switch to **Architecture** for a static walkthrough of the pipeline stages and routing table

## Notes

- Imports directly from `src/` — no logic is duplicated
- API key is masked and never displayed in plaintext
- `demo/requirements.txt` is separate from the container's `requirements.txt` — Streamlit is not in the submission image
