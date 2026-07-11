import json
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_API_ERROR_PATTERNS = re.compile(
    r"(I cannot|as an AI language model I don't have access)", re.IGNORECASE
)
_NER_KEYS = {"person", "organization", "location", "date"}

# Generous limits — only catch truly runaway responses, not verbose-but-valid answers
_CATEGORY_MAX_LENGTHS = {
    "sentiment_classification": 600,
    "math_reasoning": 400,
    "logical_reasoning": 1500,
    "factual_knowledge": 1200,
    "summarisation": 2000,
    "ner": 2000,
    "code_debugging": 4000,
    "code_generation": 5000,
}

_MIN_COMPLETENESS_CATEGORIES = {"code_debugging", "factual_knowledge", "summarisation"}
_ELABORATION_TRIGGERS = re.compile(
    r"\b(explain|why|how|describe|what causes|what is the reason)\b|```", re.IGNORECASE
)


def strip_reasoning_blocks(raw: str) -> str:
    """Discard everything up to and including closing reasoning tags, then strip leftover tags."""
    cleaned = raw.strip()
    for tag in [r"</think>", r"</thought>", r"</reason>", r"</reasoning>"]:
        match = re.search(tag, cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():].strip()
    cleaned = re.sub(r"<(?:think|thought|reason|reasoning)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(?:think|thought|reason|reasoning)>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def check_min_completeness(category: str, answer: str, prompt: str) -> Optional[str]:
    """
    Returns a retry instruction if the answer is suspiciously short for a category
    that expects an explanation. Returns None if the answer looks fine.
    """
    if category not in _MIN_COMPLETENESS_CATEGORIES:
        return None
    if len(answer.split()) >= 5:
        return None
    if _ELABORATION_TRIGGERS.search(prompt):
        return ("Your answer must fully address the question, including a brief explanation "
                "— not just a short phrase.")
    return None


def extract_last_resort(raw: str, category: str) -> str:
    """Last-resort extraction from a response that may contain leaked reasoning."""
    if not raw or not raw.strip():
        return raw

    text = strip_reasoning_blocks(raw)

    if category == "sentiment_classification":
        idx = text.lower().rfind("sentiment:")
        if idx != -1:
            return text[idx:].strip()

    elif category == "math_reasoning":
        idx = text.lower().rfind("answer:")
        if idx != -1:
            return text[idx:].strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in reversed(lines):
            if re.search(r"\d", line):
                return line

    elif category == "ner":
        match = re.findall(r"(\{[\s\S]*?\})", text)
        if match:
            return match[-1].strip()

    elif category in ("code_generation", "code_debugging"):
        match = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
        if match:
            return match[-1].strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and len(lines[-1]) < 150:
        return lines[-1]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        return sentences[-1].strip()

    return text


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

    # Step 1: Strip reasoning blocks (<think>...</think> etc.)
    raw = strip_reasoning_blocks(raw_answer)

    # Step 2: Check for standard API refusals
    if len(raw) < 50 and _API_ERROR_PATTERNS.search(raw):
        return False, None, "Response looks like an API refusal"

    # Step 3: Check length (phrase-based leak check removed — too many false positives)
    max_len = _CATEGORY_MAX_LENGTHS.get(category, 2000)
    if len(raw) > max_len:
        return False, None, f"Answer too long for {category} ({len(raw)} > {max_len} chars)"

    # Step 4: Category-specific validations
    if category == "ner":
        return _validate_ner(raw)
    if category == "sentiment_classification":
        return _validate_sentiment(raw)
    if category in ("code_generation", "code_debugging"):
        return _validate_code(raw)
    if category == "math_reasoning":
        return _validate_math(raw)

    return True, raw, None
