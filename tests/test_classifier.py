import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.classifier import classify_task


@pytest.mark.parametrize("prompt,expected", [
    # Basic happy-path (one per category)
    ("What is the capital of France?", "factual_knowledge"),
    ("If a train travels 60 mph for 2.5 hours, how many miles does it travel?", "math_reasoning"),
    ("Classify the sentiment of this review: 'I loved it!'", "sentiment_classification"),
    ("Summarise the following text in one sentence: 'The Amazon rainforest...'", "summarisation"),
    ("Extract all named entities from: 'Elon Musk founded SpaceX in 2002.'", "ner"),
    ("The following code has a bug, fix it:\n```python\ndef add(a,b): return a-b\n```\nThe function should add but gives wrong result.", "code_debugging"),
    ("A, B, and C each have a different pet. Exactly one of them has a dog.", "logical_reasoning"),
    ("Write a function that returns the sum of all even numbers in a list.", "code_generation"),
    # Edge cases
    ("2+2", "math_reasoning"),
    ("Calculate the total if x=5 and y=10", "math_reasoning"),
    ("Is this positive or negative? 'Great product!'", "sentiment_classification"),
    ("implement a binary search algorithm", "code_generation"),
    ("identify all persons and locations in: 'Obama visited Paris'", "ner"),
    ("tl;dr: long article text here...", "summarisation"),
    ("condense this to 3 bullet points: some text", "summarisation"),
    ("if and only if A is true, B must be true. A is false. What about B?", "logical_reasoning"),
])
def test_classify(prompt, expected):
    assert classify_task(prompt, "test") == expected


def test_empty_prompt_defaults_to_factual():
    assert classify_task("", "empty") == "factual_knowledge"


def test_ambiguous_logs_warning(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="src.classifier"):
        result = classify_task(
            "Summarise this text and extract the named entities from it: 'Apple Inc. was founded by Steve Jobs.'",
            "ambiguous"
        )
    # Should pick first matching rule (summarisation before ner in priority)
    assert result in ("summarisation", "ner")
