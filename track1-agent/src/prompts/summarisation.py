def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Summarise the following text according to the exact length/format instruction given. "
            "Do not exceed the specified constraint."
        )},
        {"role": "user", "content": task_prompt},
    ]
