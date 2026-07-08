def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": "Answer the question directly and concisely. Do not repeat the question. No preamble."},
        {"role": "user", "content": task_prompt},
    ]
