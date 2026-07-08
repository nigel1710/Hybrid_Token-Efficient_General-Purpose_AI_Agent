import re
from typing import Callable


# ---------------------------------------------------------------------------
# Factual knowledge
# ---------------------------------------------------------------------------

def build_factual_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": "Answer the question directly and concisely. Do not repeat the question. No preamble."},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Mathematical reasoning
# ---------------------------------------------------------------------------

def build_math_reasoning_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Solve the problem. You may reason internally, but output ONLY the final answer "
            "clearly on the last line, prefixed with 'Answer: '."
        )},
        {"role": "user", "content": task_prompt},
    ]


def extract_math_answer(raw: str) -> str:
    match = re.search(r"Answer:\s*(.+)", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Sentiment classification
# ---------------------------------------------------------------------------

def build_sentiment_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Classify the sentiment of the text as exactly one of: positive, negative, neutral. "
            "Then give a one-sentence justification. "
            "Format: 'Sentiment: <label>. Justification: <reason>'"
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Text summarisation
# ---------------------------------------------------------------------------

def build_summarisation_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Summarise the following text according to the exact length/format instruction given. Output ONLY the summary, no reasoning, no preamble, no explanation."
            "Do not exceed the specified constraint."
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Named entity recognition
# ---------------------------------------------------------------------------

def build_ner_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            'Extract all named entities from the text. Return ONLY valid JSON with this exact structure: '
            '{"person": [...], "organization": [...], "location": [...], "date": [...]}. '
            "Use empty arrays for categories with no matches. Do not include any text outside the JSON object."
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Code debugging
# ---------------------------------------------------------------------------

def build_code_debugging_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "The following code has a bug. Identify it in one short sentence, "
            "then provide the corrected code in a single code block. "
            "Preserve the original structure and style; change only what's necessary to fix the bug."
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Logical / deductive reasoning
# ---------------------------------------------------------------------------

def build_logical_reasoning_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Solve this logic puzzle. Reason through all constraints internally, "
            "then output only the final answer(s) that satisfy every condition, clearly and concisely."
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def build_code_generation_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Write a correct, well-structured function based on the specification. "
            "Return ONLY the code in a single code block, no explanation text before or after."
        )},
        {"role": "user", "content": task_prompt},
    ]


def extract_code(raw: str) -> str:
    match = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PROMPT_BUILDERS: dict[str, Callable[[str], list]] = {
    "factual_knowledge":        build_factual_prompt,
    "math_reasoning":           build_math_reasoning_prompt,
    "sentiment_classification": build_sentiment_prompt,
    "summarisation":            build_summarisation_prompt,
    "ner":                      build_ner_prompt,
    "code_debugging":           build_code_debugging_prompt,
    "logical_reasoning":        build_logical_reasoning_prompt,
    "code_generation":          build_code_generation_prompt,
}


def get_prompt_builder(category: str) -> Callable[[str], list]:
    return _PROMPT_BUILDERS.get(category, build_factual_prompt)
