"""
End-to-end test with a mocked Fireworks client.
Verifies that results.json is always fully valid regardless of API behavior.
"""
import sys
import os
import json
import asyncio
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch environment before importing config-dependent modules
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference")
os.environ.setdefault("ALLOWED_MODELS", "accounts/fireworks/models/llama-v3p1-8b-instruct")

from src.config import load_config
from src.model_router import init_router
from src import io_handler


EDGE_TASKS_PATH = os.path.join(os.path.dirname(__file__), "sample_tasks", "tasks_edge_cases.json")


def _make_mock_client(behavior="success"):
    """Returns a coroutine that simulates different Fireworks behaviors."""
    from src.fireworks_client import CallResult

    async def mock_call(config, model, messages, category="", deadline=None, max_tokens=1024):
        if behavior == "success":
            return CallResult(success=True, content="Mock answer for testing.", usage={"prompt_tokens": 10, "completion_tokens": 5})
        elif behavior == "empty":
            return CallResult(success=True, content="", usage=None)
        elif behavior == "error":
            return CallResult(success=False, error="Simulated API error", content=None)
        elif behavior == "bad_ner":
            return CallResult(success=True, content="not valid json", usage=None)
        elif behavior == "timeout":
            return CallResult(success=False, error="Timeout", content=None)
    return mock_call


@pytest.fixture
def tmp_output(tmp_path):
    output_file = tmp_path / "results.json"
    original = io_handler.OUTPUT_PATH
    io_handler.OUTPUT_PATH = str(output_file)
    yield str(output_file)
    io_handler.OUTPUT_PATH = original


def _load_edge_tasks():
    with open(EDGE_TASKS_PATH) as f:
        return json.load(f)


def _run_pipeline_with_mock(tasks, mock_fn, output_path):
    """Run the processing pipeline with a mocked call_fireworks."""
    import src.main as main_module
    import src.fireworks_client as fc_module

    original_call_fc = fc_module.call_fireworks
    original_call_main = main_module.call_fireworks
    fc_module.call_fireworks = mock_fn
    main_module.call_fireworks = mock_fn  # main imports by name, must patch there too

    config = load_config()
    init_router(config.allowed_models)

    try:
        results = asyncio.run(main_module._run(config, tasks))
    finally:
        fc_module.call_fireworks = original_call_fc
        main_module.call_fireworks = original_call_main

    # Write results
    io_handler.OUTPUT_PATH = output_path
    io_handler.write_results(results)
    return results



def test_all_task_ids_present_on_success(tmp_output):
    tasks = _load_edge_tasks()
    results = _run_pipeline_with_mock(tasks, _make_mock_client("success"), tmp_output)

    with open(tmp_output) as f:
        output = json.load(f)

    input_ids = [t["task_id"] for t in tasks]
    output_ids = [r["task_id"] for r in output]

    for tid in input_ids:
        assert tid in output_ids, f"Missing task_id {tid} in output"


def test_all_task_ids_present_on_api_error(tmp_output):
    tasks = _load_edge_tasks()
    results = _run_pipeline_with_mock(tasks, _make_mock_client("error"), tmp_output)

    with open(tmp_output) as f:
        output = json.load(f)

    input_ids = [t["task_id"] for t in tasks]
    for tid in input_ids:
        assert any(r["task_id"] == tid for r in output), f"Missing task_id {tid}"
    # All answers must be non-empty strings
    for r in output:
        assert isinstance(r["answer"], str) and len(r["answer"]) > 0


def test_output_is_valid_json(tmp_output):
    tasks = _load_edge_tasks()
    _run_pipeline_with_mock(tasks, _make_mock_client("success"), tmp_output)

    with open(tmp_output) as f:
        raw = f.read()
    data = json.loads(raw)  # must not raise
    assert isinstance(data, list)


def test_empty_task_list(tmp_output):
    tasks = []
    io_handler.OUTPUT_PATH = tmp_output
    io_handler.write_results([])

    with open(tmp_output) as f:
        data = json.load(f)
    assert data == []


def test_answers_have_correct_keys(tmp_output):
    tasks = _load_edge_tasks()
    _run_pipeline_with_mock(tasks, _make_mock_client("success"), tmp_output)

    with open(tmp_output) as f:
        output = json.load(f)

    for r in output:
        assert set(r.keys()) == {"task_id", "answer"}


def test_empty_prompt_skips_api_and_uses_placeholder(tmp_output):
    """Tasks with an empty prompt must get a non-empty placeholder answer
    without ever calling the Fireworks API."""
    call_count = [0]

    from src.fireworks_client import CallResult

    async def counting_mock(config, model, messages, category="", deadline=None, max_tokens=1024):
        call_count[0] += 1
        return CallResult(success=True, content="should not appear", usage=None)

    tasks = [
        {"task_id": "empty_1", "prompt": ""},
        {"task_id": "empty_2", "prompt": "   "},
        {"task_id": "normal_1", "prompt": "What is 2+2?"},
    ]
    _run_pipeline_with_mock(tasks, counting_mock, tmp_output)

    with open(tmp_output) as f:
        output = json.load(f)

    output_by_id = {r["task_id"]: r["answer"] for r in output}

    # All three task_ids must appear in output
    assert "empty_1" in output_by_id
    assert "empty_2" in output_by_id
    assert "normal_1" in output_by_id

    # Empty-prompt tasks must have a non-empty placeholder (not the mock answer)
    assert output_by_id["empty_1"] and output_by_id["empty_1"] != "should not appear"
    assert output_by_id["empty_2"] and output_by_id["empty_2"] != "should not appear"

    # API was called exactly once — only for the normal task
    assert call_count[0] == 1

