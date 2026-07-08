import re


def build_prompt(task_prompt: str) -> list:
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
