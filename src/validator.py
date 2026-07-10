import json
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_API_ERROR_PATTERNS = re.compile(
    r"(I cannot|as an AI language model I don't have access)", re.IGNORECASE
)
_NER_KEYS = {"person", "organization", "location", "date"}

# Maximum allowed lengths (approx. 3x the expected clean answer length)
_CATEGORY_MAX_LENGTHS = {
    "sentiment_classification": 300,
    "math_reasoning": 100,
    "logical_reasoning": 200,
    "factual_knowledge": 400,
    "summarisation": 1200,
    "ner": 1500,
    "code_debugging": 3000,
    "code_generation": 4000,
}

# Substrings indicating that internal reasoning monologue has leaked into the answer
_REASONING_LEAK_PHRASES = [
    "we need to",
    "let's",
    "let us",
    "hmm",
    "the user wants",
    "i need to figure out",
]


def strip_reasoning_blocks(raw: str) -> str:
    """
    Scans the response for closing tags like </think>, </thought>, etc.
    If present, it discards everything up to and including that closing tag,
    keeping only what follows. Otherwise, strips any individual tags.
    """
    cleaned = raw.strip()
    
    # Check for closing tags and discard everything before/including them
    closing_tags = [r"</think>", r"</thought>", r"</reason>", r"</reasoning>"]
    for tag in closing_tags:
        match = re.search(tag, cleaned, re.IGNORECASE)
        if match:
            # Keep only what follows the closing tag
            cleaned = cleaned[match.end():].strip()
            
    # Strip any remaining standalone reasoning tags (like opening tags <think> or leftover unclosed tags)
    cleaned = re.sub(r"<(?:think|thought|reason|reasoning)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(?:think|thought|reason|reasoning)>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def contains_reasoning_leak(text: str) -> bool:
    """
    Checks if the response contains typical thinking monologue phrases.
    """
    lowered = text.lower()
    for phrase in _REASONING_LEAK_PHRASES:
        if phrase in lowered:
            return True
    return False


def extract_last_resort(raw: str, category: str) -> str:
    """
    Last-resort extraction: extracts just the final sentence or labeled portion
    from a response that has leaked reasoning monologue.
    """
    if not raw or not raw.strip():
        return raw
        
    text = strip_reasoning_blocks(raw)
    
    # 1. Category-specific extractions
    if category == "sentiment_classification":
        # Extract starting from the last occurrence of "Sentiment:"
        idx = text.lower().rfind("sentiment:")
        if idx != -1:
            return text[idx:].strip()
            
    elif category == "math_reasoning":
        # Extract starting from the last occurrence of "Answer:"
        idx = text.lower().rfind("answer:")
        if idx != -1:
            return text[idx:].strip()
        # Fallback to the last line containing a digit/number
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in reversed(lines):
            if re.search(r"\d", line):
                return line
                
    elif category == "ner":
        # Extract the last JSON-like block
        match = re.findall(r"(\{[\s\S]*?\})", text)
        if match:
            return match[-1].strip()
            
    elif category in ("code_generation", "code_debugging"):
        # Extract the last code block
        match = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
        if match:
            return match[-1].strip()

    # 2. General fallback: if multiple lines, try using the last line if it is short
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and len(lines[-1]) < 150:
        return lines[-1]
        
    # Split into sentences and return the last sentence
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

    # Step 3: Check for reasoning leaks (phrases)
    if contains_reasoning_leak(raw):
        return False, None, "Reasoning leak: contains thinking trace monologue"

    # Step 4: Check for length constraints
    max_len = _CATEGORY_MAX_LENGTHS.get(category, 1000)
    if len(raw) > max_len:
        return False, None, f"Reasoning leak: answer exceeds expected length constraint for {category} ({len(raw)} > {max_len} chars)"

    # Step 5: Category-specific validations
    if category == "ner":
        return _validate_ner(raw)
    if category == "sentiment_classification":
        return _validate_sentiment(raw)
    if category in ("code_generation", "code_debugging"):
        return _validate_code(raw)
    if category == "math_reasoning":
        return _validate_math(raw)

    return True, raw, None
