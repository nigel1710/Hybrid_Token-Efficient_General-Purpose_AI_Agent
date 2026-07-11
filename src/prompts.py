import re
from typing import Callable


# ---------------------------------------------------------------------------
# Factual knowledge
# ---------------------------------------------------------------------------

def build_factual_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Answer the question directly and concisely. Do not repeat the question. No preamble. "
            "Answer only what is asked. Do not add supplementary facts, examples, or trivia "
            "beyond what directly answers the question, even if related."
        )},
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
            "Use empty arrays for categories with no matches. Do not include any text outside the JSON object. "
            "IMPORTANT type rules: "
            "Countries, nations, and nationalities (e.g. 'the United States', 'France', 'Japan') are LOCATION, not ORGANIZATION. "
            "Only actual institutions, companies, and formal bodies (e.g. 'Microsoft', 'the United Nations', 'Congress') are ORGANIZATION. "
            "Example: 'Apple was founded in the United States' → "
            '{"person": [], "organization": ["Apple"], "location": ["United States"], "date": []}.'
        )},
        {"role": "user", "content": task_prompt},
    ]


# ---------------------------------------------------------------------------
# Code debugging
# ---------------------------------------------------------------------------

def build_code_debugging_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "The following code has a bug. "
            "State your conclusion in one clear, direct, non-contradictory sentence first — "
            "either identify the exact bug, or state clearly that no bug exists for the described symptoms. "
            "Do not present an incorrect claim and then walk it back in the same response. "
            "Do not add unrelated suggestions unless they directly relate to the reported issue. "
            "Then provide the corrected code in a single code block. "
            "Preserve the original structure and style; change only what is necessary to fix the bug."
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


def build_compound_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "The user is asking you to do two things. Address both parts explicitly and in order, "
            "clearly separated by a blank line. "
            "For any summarisation part: provide a concise one-sentence summary. "
            "For any named-entity extraction part: return a JSON object with keys "
            '"person", "organization", "location", "date" (use empty arrays for no matches). '
            "Countries and nations are LOCATION. Companies and institutions are ORGANIZATION. "
            "Do not merge the two parts or omit either one."
        )},
        {"role": "user", "content": task_prompt},
    ]


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
    "compound":                 build_compound_prompt,
}


def get_prompt_builder(category: str) -> Callable[[str], list]:
    return _PROMPT_BUILDERS.get(category, build_factual_prompt)
