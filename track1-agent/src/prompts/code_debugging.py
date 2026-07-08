def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "The following code has a bug. Identify it in one short sentence, "
            "then provide the corrected code in a single code block. "
            "Preserve the original structure and style; change only what's necessary to fix the bug."
        )},
        {"role": "user", "content": task_prompt},
    ]
