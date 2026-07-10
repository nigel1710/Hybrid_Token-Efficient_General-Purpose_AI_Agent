import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/ without installing the package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier import classify_task
from src.model_router import init_router, get_model_for_category, mark_model_unavailable, CATEGORY_TIER, _MODEL_TIERS
from src.fireworks_client import call_fireworks
from src.validator import validate_and_finalize
from src.prompts import get_prompt_builder, extract_math_answer, extract_code
from src.config import Config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Overfitters — Track 1 Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_TASKS_PATH = ROOT / "tests" / "sample_tasks" / "tasks_basic.json"

CATEGORY_COLORS = {
    "factual_knowledge":        "#4C9BE8",
    "math_reasoning":           "#E8844C",
    "sentiment_classification": "#4CE8A0",
    "summarisation":            "#A04CE8",
    "ner":                      "#E8D44C",
    "code_debugging":           "#E84C4C",
    "logical_reasoning":        "#4CE8E8",
    "code_generation":          "#E84CA0",
}

TIER_BADGE = {"CHEAP": "🟢 CHEAP", "MID": "🟡 MID", "LARGE": "🔴 LARGE", "override": "⚡ CODE OVERRIDE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_model(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _build_config(api_key: str, base_url: str, allowed_models: list) -> Config:
    return Config(api_key=api_key, base_url=base_url.rstrip("/"), allowed_models=allowed_models)


def _post_process(category: str, raw: str) -> str:
    if category == "math_reasoning":
        return extract_math_answer(raw)
    if category == "code_generation":
        return extract_code(raw)
    return raw


async def _run_single_task(config: Config, task: dict) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"].strip() or "Empty prompt."
    category = classify_task(prompt, task_id)

    max_tokens = {
        "logical_reasoning": 2048, "math_reasoning": 1024,
        "code_generation": 1024, "code_debugging": 1024,
        "summarisation": 512, "ner": 512,
        "factual_knowledge": 512, "sentiment_classification": 256,
    }.get(category, 512)

    messages = get_prompt_builder(category)(prompt)
    best_raw: Optional[str] = None
    retries = 0
    fallback_used = False
    tokens_used = 0
    final_answer = ""
    status = "✅ passed"

    for attempt in range(3):
        # Re-resolve model each attempt so 404-marked models are skipped
        model = get_model_for_category(category)
        tier = CATEGORY_TIER.get(category, "MID")
        if "kimi" in model and category in ("code_debugging", "code_generation"):
            tier = "override"

        result = await call_fireworks(
            config, model, messages, category=category, max_tokens=max_tokens
        )
        if result.usage:
            tokens_used += result.usage.get("total_tokens", 0)

        if not result.success:
            retries += 1
            if "404" in (result.error or ""):
                mark_model_unavailable({model})
            continue

        raw = _post_process(category, result.content)
        is_valid, cleaned, reason = validate_and_finalize(task_id, category, raw, attempt)

        if is_valid:
            final_answer = cleaned
            if attempt > 0:
                status = f"⚠️ passed (retry {attempt})"
            break

        best_raw = raw if raw else best_raw
        retries += 1
        status = "⚠️ passed after retry"
        # append retry hint to messages
        messages = get_prompt_builder(category)(
            prompt + f"\n\n[Previous response invalid: {reason}. Please correct it.]"
        )
    else:
        fallback_used = True
        final_answer = best_raw or f"Unable to generate answer for task_id={task_id}."
        status = "🔴 fallback used"

    return {
        "task_id": task_id,
        "prompt": prompt,
        "category": category,
        "tier": tier,
        "model": model,
        "short_model": _short_model(model),
        "tokens_used": tokens_used,
        "retries": retries,
        "fallback_used": fallback_used,
        "status": status,
        "answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/robot-2.png", width=64)
st.sidebar.title("The Overfitters")
st.sidebar.caption("Track 1 — General-Purpose AI Agent")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["🚀 Run Agent", "📊 Results Dashboard", "🏗️ Architecture"])
st.sidebar.divider()
st.sidebar.caption("Demo only — not part of the Docker submission image.")

# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False


# ===========================================================================
# PAGE 1 — Run Agent
# ===========================================================================
if page == "🚀 Run Agent":
    st.title("🤖 The Overfitters — Track 1 Agent")
    st.caption(
        "This demo calls the **same classification, routing, and validation code** as our actual "
        "Docker submission container — just wrapped in a UI for demonstration purposes."
    )
    st.divider()

    # --- Config panel ---
    with st.expander("⚙️ API Configuration", expanded=not bool(os.environ.get("FIREWORKS_API_KEY"))):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "FIREWORKS_API_KEY",
                value=os.environ.get("FIREWORKS_API_KEY", ""),
                type="password",
                help="Never displayed in plaintext after entry.",
            )
            base_url = st.text_input(
                "FIREWORKS_BASE_URL",
                value=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference"),
            )
        with col2:
            default_models = os.environ.get(
                "ALLOWED_MODELS",
                "accounts/fireworks/models/minimax-m3,"
                "accounts/fireworks/models/kimi-k2p7-code,"
                "accounts/fireworks/models/gemma-4-31b-it,"
                "accounts/fireworks/models/gemma-4-26b-a4b-it,"
                "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
            )
            allowed_models_raw = st.text_area(
                "ALLOWED_MODELS (comma-separated)",
                value=default_models,
                height=120,
            )

    allowed_models = [m.strip() for m in allowed_models_raw.split(",") if m.strip()]

    # --- Task input ---
    # tasks are stored in session_state so they survive button-click reruns
    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []

    st.subheader("📋 Tasks Input")
    input_tab, paste_tab = st.tabs(["📁 Upload tasks.json", "✏️ Paste / Edit JSON"])

    with input_tab:
        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded = st.file_uploader("Upload tasks.json", type="json", label_visibility="collapsed")
            if uploaded:
                try:
                    loaded = json.loads(uploaded.read())
                    st.session_state["tasks"] = loaded
                    st.success(f"Loaded {len(loaded)} tasks from uploaded file.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        with col_sample:
            if st.button("📂 Load sample batch", use_container_width=True):
                if SAMPLE_TASKS_PATH.exists():
                    loaded = json.loads(SAMPLE_TASKS_PATH.read_text(encoding="utf-8"))
                    st.session_state["tasks"] = loaded
                    st.session_state["pasted_json"] = json.dumps(loaded, indent=2)
                    st.success(f"Loaded {len(loaded)} sample tasks.")
                else:
                    st.warning("Sample file not found.")

    with paste_tab:
        default_paste = st.session_state.get("pasted_json", '[{"task_id": "t01", "prompt": "What is the capital of France?"}]')
        pasted = st.text_area("Paste JSON array of tasks", value=default_paste, height=200, key="paste_area")
        if st.button("Load pasted JSON", key="load_paste"):
            try:
                parsed = json.loads(pasted)
                st.session_state["tasks"] = parsed
                st.session_state["pasted_json"] = pasted
                st.success(f"Loaded {len(parsed)} tasks.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    tasks = st.session_state["tasks"]
    if tasks:
        st.info(f"**{len(tasks)} task(s)** ready to process.")

    st.divider()

    # --- Run button ---
    run_disabled = not (api_key and base_url and allowed_models and tasks)
    if not api_key:
        st.warning("Enter your FIREWORKS_API_KEY in the config panel above to enable the Run button.")

    if st.button("▶️ Run Agent", type="primary", disabled=run_disabled, use_container_width=True):
        try:
            config = _build_config(api_key, base_url, allowed_models)
            init_router(allowed_models)
        except Exception as e:
            st.error(f"Config error: {e}")
            st.stop()

        st.session_state.results = []
        st.session_state.run_complete = False
        results_accumulator = []

        progress = st.progress(0, text="Starting...")
        live_table_placeholder = st.empty()

        for i, task in enumerate(tasks):
            progress.progress((i) / len(tasks), text=f"Processing task {i+1}/{len(tasks)}: `{task['task_id']}`")
            with st.status(f"Task `{task['task_id']}` — {task['prompt'][:80]}...", expanded=False) as task_status:
                t0 = time.monotonic()
                try:
                    result = asyncio.run(_run_single_task(config, task))
                except Exception as e:
                    result = {
                        "task_id": task["task_id"], "prompt": task["prompt"],
                        "category": "error", "tier": "-", "model": "-", "short_model": "-",
                        "tokens_used": 0, "retries": 0, "fallback_used": True,
                        "status": "🔴 error", "answer": str(e),
                    }
                elapsed = time.monotonic() - t0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Category", result["category"].replace("_", " ").title())
                col2.metric("Model", result["short_model"])
                col3.metric("Tokens", result["tokens_used"])
                col4.metric("Latency", f"{elapsed:.1f}s")

                st.write(f"**Status:** {result['status']}  |  **Retries:** {result['retries']}")
                st.write("**Answer:**")
                st.code(result["answer"][:600] + ("..." if len(result["answer"]) > 600 else ""), language=None)

                if result["fallback_used"]:
                    task_status.update(label=f"Task `{task['task_id']}` — 🔴 fallback used", state="error")
                else:
                    task_status.update(label=f"Task `{task['task_id']}` — {result['status']}", state="complete")

            results_accumulator.append(result)

            # Live table update
            df_live = pd.DataFrame(results_accumulator)[
                ["task_id", "category", "short_model", "tokens_used", "retries", "status"]
            ].rename(columns={"short_model": "model"})
            live_table_placeholder.dataframe(df_live, use_container_width=True, hide_index=True)

        progress.progress(1.0, text="All tasks complete!")
        st.session_state.results = results_accumulator
        st.session_state.run_complete = True
        st.success(f"✅ Completed {len(results_accumulator)} tasks. Switch to **Results Dashboard** to explore.")


# ===========================================================================
# PAGE 2 — Results Dashboard
# ===========================================================================
elif page == "📊 Results Dashboard":
    st.title("📊 Results Dashboard")

    if not st.session_state.run_complete or not st.session_state.results:
        st.info("No results yet. Run the agent first on the **Run Agent** page.")
        st.stop()

    results = st.session_state.results
    df = pd.DataFrame(results)

    total_tasks = len(df)
    total_tokens = int(df["tokens_used"].sum())
    avg_tokens = round(total_tokens / total_tasks, 1) if total_tasks else 0
    retry_rate = round(df["retries"].gt(0).mean() * 100, 1)
    pass_rate = round(df["fallback_used"].eq(False).mean() * 100, 1)

    # --- Summary metrics ---
    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tasks", total_tasks)
    c2.metric("Total Tokens", f"{total_tokens:,}")
    c3.metric("Avg Tokens / Task", avg_tokens)
    c4.metric("Retry Rate", f"{retry_rate}%")
    c5.metric("Pass Rate", f"{pass_rate}%")

    st.divider()

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Token Usage by Category")
        token_by_cat = df.groupby("category")["tokens_used"].sum().reset_index()
        token_by_cat.columns = ["Category", "Tokens"]
        fig1 = px.bar(
            token_by_cat, x="Category", y="Tokens",
            color="Category",
            color_discrete_map={k: v for k, v in CATEGORY_COLORS.items()},
            text="Tokens",
        )
        fig1.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("Tasks Routed per Model")
        model_counts = df["short_model"].value_counts().reset_index()
        model_counts.columns = ["Model", "Tasks"]
        fig2 = px.pie(model_counts, names="Model", values="Tasks", hole=0.4)
        fig2.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Full results table ---
    st.subheader("Full Results Table")
    display_df = df[["task_id", "category", "short_model", "tier", "tokens_used", "retries", "status", "answer"]].copy()
    display_df.columns = ["Task ID", "Category", "Model", "Tier", "Tokens", "Retries", "Status", "Answer"]
    display_df["Answer"] = display_df["Answer"].str[:120] + "..."

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="medium"),
            "Answer": st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()

    # --- Download ---
    submission = [{"task_id": r["task_id"], "answer": r["answer"]} for r in results]
    st.download_button(
        label="⬇️ Download results.json (submission format)",
        data=json.dumps(submission, ensure_ascii=False, indent=2),
        file_name="results.json",
        mime="application/json",
        type="primary",
    )


# ===========================================================================
# PAGE 3 — Architecture Explainer
# ===========================================================================
elif page == "🏗️ Architecture":
    st.title("🏗️ Pipeline Architecture")
    st.caption("How each task flows through our agent — from raw JSON to final answer.")
    st.divider()

    # Pipeline stages
    stages = [
        ("📥", "tasks.json", "Input batch of tasks, each with a task_id and prompt."),
        ("🔍", "Classifier", "Rule-based regex classifier assigns each prompt to one of 8 categories."),
        ("🗺️", "Model Router", "Maps category → tier (CHEAP/MID/LARGE) → specific Fireworks model ID."),
        ("🌐", "Fireworks API", "Async HTTP call with per-category prompt template, retries, and timeout."),
        ("✅", "Validator", "Per-category validation: NER JSON repair, sentiment label check, code heuristics."),
        ("📤", "results.json", "Atomic write of final answers in submission format."),
    ]

    cols = st.columns(len(stages))
    for col, (icon, name, desc) in zip(cols, stages):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{name}**")
            st.caption(desc)
            if name not in ("tasks.json", "results.json"):
                st.markdown("↓")

    st.divider()

    # Routing table
    st.subheader("Category → Tier → Model Routing Table")

    routing_rows = []
    for category, tier in CATEGORY_TIER.items():
        if category in ("code_debugging", "code_generation"):
            model_id = "accounts/fireworks/models/kimi-k2p7-code"
            effective_tier = "CODE OVERRIDE"
        else:
            model_id = next(
                (m for m, t in _MODEL_TIERS.items() if t == tier), tier
            )
            effective_tier = tier
        routing_rows.append({
            "Category": category.replace("_", " ").title(),
            "Tier": effective_tier,
            "Model": _short_model(model_id),
            "Rationale": {
                "CHEAP": "Fast, low-cost — sufficient for simple classification/extraction",
                "MID": "Balanced accuracy/cost for general tasks",
                "LARGE": "Strongest reasoning — reserved for math and logic",
                "CODE OVERRIDE": "Code-specialized model regardless of tier",
            }.get(effective_tier, ""),
        })

    st.table(pd.DataFrame(routing_rows))

    st.divider()

    # Model tier legend
    st.subheader("Model Tier Assignments")
    tier_rows = [
        {"Model": _short_model(m), "Tier": t, "Full ID": m}
        for m, t in _MODEL_TIERS.items()
    ]
    st.table(pd.DataFrame(tier_rows))

    st.divider()
    st.subheader("Key Design Decisions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Token Efficiency**")
        st.caption(
            "Cheap Gemma MoE models handle simple tasks (sentiment, NER, factual). "
            "Expensive models (minimax-m3) are reserved only for math and logic where they matter."
        )
    with col2:
        st.markdown("**Reliability**")
        st.caption(
            "Gemma on-demand deployments can cold-start (404). The client detects this and retries "
            "with longer backoff. A warm-up step pre-spins models before the task clock starts."
        )
    with col3:
        st.markdown("**Validation**")
        st.caption(
            "Each category has a tailored validator: NER auto-repairs missing JSON keys, "
            "math checks for digits, code checks for code content. Reasoning leaks are stripped."
        )
