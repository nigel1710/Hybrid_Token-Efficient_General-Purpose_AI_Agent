import re
import logging

logger = logging.getLogger(__name__)

CATEGORIES = [
    "factual_knowledge",
    "math_reasoning",
    "sentiment_classification",
    "summarisation",
    "ner",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]


def _has_code_block(text: str) -> bool:
    return bool(re.search(r"```|^\s{4,}\S", text, re.MULTILINE))


def _has_code_keywords(text: str) -> bool:
    return bool(re.search(r"\bdef \b|\bfunction\b|\bclass \b|\bimport \b|#include", text))


def _looks_like_code_debugging(text: str) -> bool:
    has_code = _has_code_block(text) or _has_code_keywords(text)
    has_bug = bool(re.search(
        r"\bbug\b|\berror\b|\bfix\b|\bdebug\b|not working|incorrect output|"
        r"\bexception\b|\btraceback\b|\bfails?\b|\bbroken\b",
        text, re.IGNORECASE
    ))
    return has_code and has_bug


def _looks_like_code_generation(text: str) -> bool:
    return bool(re.search(
        r"write a function|implement\b|write code|create a program|write a script|"
        r"write a class|generate a function|code that\b",
        text, re.IGNORECASE
    ))


def _looks_like_ner(text: str) -> bool:
    return bool(re.search(
        r"extract (all )?(named )?entities|named entities|identify all\b|"
        r"\b(person|organization|location|date)\b.{0,40}\bextract\b|"
        r"\bextract\b.{0,40}\b(person|organization|location|date)\b",
        text, re.IGNORECASE
    ))


def _looks_like_summarisation(text: str) -> bool:
    return bool(re.search(
        r"\bsummar[iy]s[ei]\b|\bcondense\b|\btl;?dr\b|\bshorten\b|"
        r"in one sentence|in \d+ (bullet|sentence|word)|under \d+ word",
        text, re.IGNORECASE
    ))


def _looks_like_sentiment(text: str) -> bool:
    return bool(re.search(
        r"\bsentiment\b|classify the tone|positive or negative|"
        r"how does the reviewer feel|is this (review )?(positive|negative|neutral)",
        text, re.IGNORECASE
    ))


def _looks_like_math(text: str) -> bool:
    has_numbers = bool(re.search(r"\d", text))
    has_math_words = bool(re.search(
        r"\bcalculate\b|\bhow many\b|\btotal\b|\baverage\b|\bpercent\b|"
        r"\bsum\b|\bproduct\b|\bdivide\b|\bmultiply\b|\bsolve\b|"
        r"[+\-*/=]|\b\d+\s*[\+\-\*\/]\s*\d+",
        text, re.IGNORECASE
    ))
    return has_numbers and has_math_words


def _looks_like_logical_reasoning(text: str) -> bool:
    return bool(re.search(
        r"exactly one of|must be true|if and only if|either.{1,20}or\b|"
        r"\beach of\b|\bnone of\b|\ball of\b|"
        r"(A|B|C|D),?\s+(B|C|D|E),?\s+(and|or)\s+(C|D|E)\s+(each|all|must|can)",
        text, re.IGNORECASE
    ))


_RULES = [
    ("code_debugging", _looks_like_code_debugging),
    ("code_generation", _looks_like_code_generation),
    ("ner", _looks_like_ner),
    ("summarisation", _looks_like_summarisation),
    ("sentiment_classification", _looks_like_sentiment),
    ("math_reasoning", _looks_like_math),
    ("logical_reasoning", _looks_like_logical_reasoning),
]


_COMPOUND_PATTERN = re.compile(
    r"(summar\w+|extract|classify|identify|list|explain|describe)"
    r".{1,60}\band\b.{1,60}"
    r"(summar\w+|extract|classify|identify|list|explain|describe)",
    re.IGNORECASE | re.DOTALL,
)


def is_compound_task(prompt: str) -> bool:
    """Returns True if the prompt contains two distinct category-trigger verbs joined by 'and'."""
    return bool(_COMPOUND_PATTERN.search(prompt.strip()))


def classify_task(prompt: str, task_id: str = "") -> str:
    text = prompt.strip()

    # Compound prompts get their own category so main.py can build a combined prompt
    if is_compound_task(text):
        logger.info("Task %r classified as 'compound'", task_id)
        return "compound"

    matched = []
    for category, fn in _RULES:
        if fn(text):
            matched.append(category)

    if not matched:
        result = "factual_knowledge"
    else:
        result = matched[0]
        if len(matched) > 1:
            logger.warning("Task %r matches multiple categories %s — using %r (highest priority)",
                           task_id, matched, result)

    logger.info("Task %r classified as %r", task_id, result)
    return result
