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


def test_strip_reasoning_blocks():
    from src.validator import strip_reasoning_blocks
    raw1 = "<think>We need to do X</think> The final answer is Paris."
    assert strip_reasoning_blocks(raw1) == "The final answer is Paris."
    
    raw2 = "<thought>Let's think.</thought><think>another thought</think>Some other text"
    assert strip_reasoning_blocks(raw2) == "Some other text"
    
    raw3 = "Leftover tag <think>"
    assert strip_reasoning_blocks(raw3) == "Leftover tag"


def test_contains_reasoning_leak_detection():
    from src.validator import contains_reasoning_leak
    assert contains_reasoning_leak("Well, we need to find the sum...")
    assert contains_reasoning_leak("Let's analyze the input text.")
    assert contains_reasoning_leak("Hmm, the user wants me to do this.")
    assert not contains_reasoning_leak("The capital of France is Paris.")


def test_category_max_lengths():
    # Sentiment classification has max length of 300
    long_sentiment = "Sentiment: positive. Justification: " + "a" * 350
    ok, cleaned, reason = validate_and_finalize("t1", "sentiment_classification", long_sentiment, 0)
    assert not ok
    assert "exceeds expected length" in reason or "leak" in reason.lower()
    
    # Short sentiment should pass
    short_sentiment = "Sentiment: positive. Justification: Good."
    ok, cleaned, reason = validate_and_finalize("t1", "sentiment_classification", short_sentiment, 0)
    assert ok


def test_extract_last_resort_logic():
    from src.validator import extract_last_resort
    # Sentiment classification last resort
    raw_sentiment = "We need to look at tone. Let's see... Sentiment: positive. Justification: Enthusastic."
    assert extract_last_resort(raw_sentiment, "sentiment_classification") == "Sentiment: positive. Justification: Enthusastic."
    
    # Math last resort
    raw_math = "Let's multiply. 2*3=6. Answer: 6"
    assert extract_last_resort(raw_math, "math_reasoning") == "Answer: 6"
    
    # NER last resort
    raw_ner = "We need to extract entities. Here: {\"person\": [\"Alice\"]}"
    assert extract_last_resort(raw_ner, "ner") == "{\"person\": [\"Alice\"]}"
    
    # General fallback - last line
    raw_gen = "Line 1\nLine 2\nFinal short line"
    assert extract_last_resort(raw_gen, "factual_knowledge") == "Final short line"

