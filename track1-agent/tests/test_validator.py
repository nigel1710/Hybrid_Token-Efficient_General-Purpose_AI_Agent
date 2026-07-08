import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.validator import validate_and_finalize


def test_empty_answer_rejected():
    ok, cleaned, reason = validate_and_finalize("t1", "factual_knowledge", "", 0)
    assert not ok
    assert cleaned is None


def test_none_answer_rejected():
    ok, cleaned, reason = validate_and_finalize("t1", "factual_knowledge", None, 0)
    assert not ok


def test_valid_factual():
    ok, cleaned, reason = validate_and_finalize("t1", "factual_knowledge", "Paris", 0)
    assert ok
    assert cleaned == "Paris"


def test_ner_valid_json():
    raw = '{"person": ["Elon Musk"], "organization": ["SpaceX"], "location": [], "date": ["2002"]}'
    ok, cleaned, reason = validate_and_finalize("t1", "ner", raw, 0)
    assert ok
    import json
    data = json.loads(cleaned)
    assert "person" in data


def test_ner_missing_keys_auto_filled():
    raw = '{"person": ["Alice"]}'
    ok, cleaned, reason = validate_and_finalize("t1", "ner", raw, 0)
    assert ok
    import json
    data = json.loads(cleaned)
    assert data["organization"] == []
    assert data["location"] == []
    assert data["date"] == []


def test_ner_with_code_fence():
    raw = '```json\n{"person": ["Bob"], "organization": [], "location": [], "date": []}\n```'
    ok, cleaned, reason = validate_and_finalize("t1", "ner", raw, 0)
    assert ok


def test_ner_invalid_json_rejected():
    ok, cleaned, reason = validate_and_finalize("t1", "ner", "not json at all", 0)
    assert not ok
    assert reason is not None


def test_sentiment_valid():
    raw = "Sentiment: positive. Justification: The reviewer expressed enthusiasm."
    ok, cleaned, reason = validate_and_finalize("t1", "sentiment_classification", raw, 0)
    assert ok


def test_sentiment_unusual_label_still_valid():
    raw = "Sentiment: mixed. Justification: Both good and bad aspects mentioned."
    ok, cleaned, reason = validate_and_finalize("t1", "sentiment_classification", raw, 0)
    assert ok  # soft check — don't hard-reject


def test_math_with_digit():
    ok, cleaned, reason = validate_and_finalize("t1", "math_reasoning", "The answer is 42", 0)
    assert ok


def test_api_refusal_rejected():
    ok, cleaned, reason = validate_and_finalize("t1", "factual_knowledge", "I cannot", 0)
    assert not ok


def test_fallback_never_empty():
    """Simulate all attempts failing — ensure we always get a non-empty string back."""
    # This tests the orchestrator's fallback logic conceptually via validator
    raw = "Some partial answer even if imperfect"
    ok, cleaned, reason = validate_and_finalize("t1", "factual_knowledge", raw, 2)
    assert ok
    assert cleaned and len(cleaned) > 0
