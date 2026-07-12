import json
import re
import logging
from typing import Optional, Tuple

from prompts import extract_math_answer

logger = logging.getLogger(__name__)

# Patterns to extract the required count from a summarisation prompt
_BULLET_COUNT_RE = re.compile(
    r"exactly\s+(\w+|\d+)\s+bullet", re.IGNORECASE
)
_SENTENCE_COUNT_RE = re.compile(
    r"exactly\s+(\w+|\d+)\s+sentence", re.IGNORECASE
)
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_API_ERROR_PATTERNS = re.compile(
    r"(I cannot|as an AI language model I don't have access)", re.IGNORECASE
)
_NER_KEYS = {"person", "organization", "location", "date"}

# Generous limits — only catch truly runaway responses, not verbose-but-valid answers
_CATEGORY_MAX_LENGTHS = {
    "sentiment_classification": 600,
    "math_reasoning": 1200,  # raised to accommodate visible step-by-step working
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


def _parse_word_or_digit(token: str) -> Optional[int]:
    """Convert a word ('three') or digit string ('3') to int, or return None."""
    try:
        return int(token)
    except ValueError:
        return _WORD_TO_NUM.get(token.lower())


def _extract_summarisation_constraints(prompt: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse the prompt for 'exactly N bullet points' and 'exactly N sentences'.
    Returns (required_bullets, required_sentences) — each is None if not found.
    """
    required_bullets = None
    required_sentences = None
    m = _BULLET_COUNT_RE.search(prompt)
    if m:
        required_bullets = _parse_word_or_digit(m.group(1))
    m = _SENTENCE_COUNT_RE.search(prompt)
    if m:
        required_sentences = _parse_word_or_digit(m.group(1))
    return required_bullets, required_sentences


def _count_bullets(text: str) -> int:
    """Count lines that look like bullet points (-, *, •, or numbered list)."""
    return sum(
        1 for line in text.splitlines()
        if re.match(r"^\s*[-*•]\s+", line) or re.match(r"^\s*\d+[.):]\s+", line)
    )


def _count_sentences(text: str) -> int:
    """Rough sentence count by splitting on sentence-ending punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return sum(1 for p in parts if p.strip())


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
        return extract_math_answer(text)

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
        if label not in ("positive", "negative", "neutral", "mixed"):
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
    """
    Extract just the final 'Answer: X' value from the model's response, discarding
    any visible step-by-step working — this keeps the stored answer clean and short
    for token efficiency, while the model is still allowed to show its work during
    generation to actually get multi-step arithmetic right.
    """
    extracted = extract_math_answer(raw)
    if not re.search(r"\d", extracted):
        logger.warning(
            "Extracted math answer contains no digits — may be non-numeric answer: %r",
            extracted,
        )
    return True, extracted, None


def _validate_summarisation(
    task_id: str,
    raw: str,
    prompt: str,
    attempt_number: int,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate bullet-point or sentence count constraints for summarisation tasks."""
    required_bullets, required_sentences = _extract_summarisation_constraints(prompt)

    if required_bullets is not None:
        actual_bullets = _count_bullets(raw)
        retry = actual_bullets != required_bullets
        logger.debug(
            "[summarisation] task=%s attempt=%d | "
            "required_bullets=%d actual_bullets=%d retry_triggered=%s",
            task_id, attempt_number, required_bullets, actual_bullets, retry,
        )
        if retry:
            reason = (
                f"Expected exactly {required_bullets} bullet point(s) "
                f"but got {actual_bullets}. "
                "Use a '-' character at the start of each bullet line."
            )
            return False, None, reason

    elif required_sentences is not None:
        actual_sentences = _count_sentences(raw)
        retry = actual_sentences != required_sentences
        logger.debug(
            "[summarisation] task=%s attempt=%d | "
            "required_sentences=%d actual_sentences=%d retry_triggered=%s",
            task_id, attempt_number, required_sentences, actual_sentences, retry,
        )
        if retry:
            reason = (
                f"Expected exactly {required_sentences} sentence(s) "
                f"but got {actual_sentences}."
            )
            return False, None, reason
    else:
        logger.debug(
            "[summarisation] task=%s attempt=%d | "
            "no bullet/sentence constraint detected in prompt — accepting as-is",
            task_id, attempt_number,
        )

    return True, raw, None


def validate_and_finalize(
    task_id: str,
    category: str,
    raw_answer: Optional[str],
    attempt_number: int,
    prompt: str = "",
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
    if category == "summarisation":
        return _validate_summarisation(task_id, raw, prompt, attempt_number)

    return True, raw, None