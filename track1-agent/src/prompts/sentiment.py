def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            "Classify the sentiment of the text as exactly one of: positive, negative, neutral. "
            "Then give a one-sentence justification. "
            "Format: 'Sentiment: <label>. Justification: <reason>'"
        )},
        {"role": "user", "content": task_prompt},
    ]
