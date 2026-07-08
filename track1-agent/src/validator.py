import json
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_API_ERROR_PATTERNS = re.compile(
    r"(I cannot|as an AI language model I don't have access)", re.IGNORECASE
)
_NER_KEYS = {"person", "organization", "location", "date"}


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_ner(raw: str) -> Tuple[bool, Optional[str], Optional[str]]:
    for attempt_text in [raw, _strip_code_fence(raw)]:
        try:
            data = json.loads(attempt_text)
            if isinstance(data, dict):
                for k in _NER_KEYS:
                    if k not in data:
                        data[k] = []
                return True, json.dumps(data), None
        except json.JSONDecodeError:
            pass
    return False, None, "NER output is not valid JSON"


def _validate_sentiment(raw: str) -> Tuple[bool, Optional[str], Optional[str]]:
    match = re.search(r"Sentiment:\s*(\w+)", raw, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        if label not in ("positive", "negative", "neutral"):
            logger.warning("Sentiment label %r not in expected set — keeping raw answer", label)
    else:
        logger.warning("No 'Sentiment:' label found in response — keeping raw answer")
    return True, raw, None


def _validate_code(raw: str) -> Tuple[bool, Optional[str], Optional[str]]:
    has_code = bool(re.search(r"def |function |class |\{|;|```", raw))
    if not has_code:
        logger.warning("Code answer may be missing code content")
    return True, raw, None


def _validate_math(raw: str) -> Tuple[bool, Optional[str], Optional[str]]:
    if not re.search(r"\d", raw):
        logger.warning("Math answer contains no digits — may be non-numeric answer")
    return True, raw, None


def validate_and_finalize(
    task_id: str,
    category: str,
    raw_answer: Optional[str],
    attempt_number: int,
) -> Tuple[bool, Optional[str], Optional[str]]:
    if not raw_answer or not raw_answer.strip():
        return False, None, "Empty or None answer"

    raw = raw_answer.strip()

    if len(raw) < 50 and _API_ERROR_PATTERNS.search(raw):
        return False, None, "Response looks like an API refusal"

    if category == "ner":
        return _validate_ner(raw)
    if category == "sentiment_classification":
        return _validate_sentiment(raw)
    if category in ("code_generation", "code_debugging"):
        return _validate_code(raw)
    if category == "math_reasoning":
        return _validate_math(raw)

    return True, raw, None
