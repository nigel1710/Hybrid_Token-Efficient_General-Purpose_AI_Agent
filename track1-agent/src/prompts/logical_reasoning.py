def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Solve this logic puzzle. Reason through all constraints internally, "
            "then output only the final answer(s) that satisfy every condition, clearly and concisely."
        )},
        {"role": "user", "content": task_prompt},
    ]
