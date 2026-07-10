import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import io_handler


@pytest.fixture
def temp_output_file(tmp_path):
    output_file = tmp_path / "results_roundtrip.json"
    original = io_handler.OUTPUT_PATH
    io_handler.OUTPUT_PATH = str(output_file)
    yield str(output_file)
    io_handler.OUTPUT_PATH = original
    if os.path.exists(str(output_file)):
        os.remove(str(output_file))


@pytest.fixture
def temp_input_file(tmp_path):
    input_file = tmp_path / "tasks_input.json"
    original = io_handler.TASKS_PATH
    io_handler.TASKS_PATH = str(input_file)
    yield str(input_file)
    io_handler.TASKS_PATH = original
    if os.path.exists(str(input_file)):
        os.remove(str(input_file))


def test_write_read_roundtrip(temp_output_file):
    test_results = [
        {"task_id": "1", "answer": "Answer 1"},
        {"task_id": "2", "answer": "Answer 2"},
        {"task_id": "3", "answer": "Answer 3"},
    ]

    # Write results using io_handler
    io_handler.write_results(test_results)

    # Assert that the file exists at the absolute path
    assert os.path.exists(temp_output_file)

    # Read file directly and check content
    with open(temp_output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["task_id"] == "1"
    assert data[0]["answer"] == "Answer 1"
    assert data[1]["task_id"] == "2"
    assert data[1]["answer"] == "Answer 2"
    assert data[2]["task_id"] == "3"
    assert data[2]["answer"] == "Answer 3"


def test_write_empty_roundtrip(temp_output_file):
    # Write empty results
    io_handler.write_results([])

    assert os.path.exists(temp_output_file)

    with open(temp_output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data == []


def test_read_tasks_filters_duplicates(temp_input_file):
    # Setup tasks JSON with duplicate task_ids
    input_data = [
        {"task_id": "task_1", "prompt": "Prompt 1"},
        {"task_id": "task_2", "prompt": "Prompt 2"},
        {"task_id": "task_1", "prompt": "Duplicate Prompt 1 (should be skipped)"},
        {"task_id": "task_3", "prompt": "Prompt 3"},
        {"task_id": "task_2", "prompt": "Duplicate Prompt 2 (should be skipped)"},
    ]

    with open(temp_input_file, "w", encoding="utf-8") as f:
        json.dump(input_data, f)

    # Read tasks
    tasks = io_handler.read_tasks()

    # Assert that only unique, first-occurring tasks are present
    assert len(tasks) == 3
    assert tasks[0] == {"task_id": "task_1", "prompt": "Prompt 1"}
    assert tasks[1] == {"task_id": "task_2", "prompt": "Prompt 2"}
    assert tasks[2] == {"task_id": "task_3", "prompt": "Prompt 3"}
