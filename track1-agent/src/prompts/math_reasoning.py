import re


def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Solve the problem. You may reason internally, but output ONLY the final answer "
            "clearly on the last line, prefixed with 'Answer: '."
        )},
        {"role": "user", "content": task_prompt},
    ]


def extract_answer(raw: str) -> str:
    match = re.search(r"Answer:\s*(.+)", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()
