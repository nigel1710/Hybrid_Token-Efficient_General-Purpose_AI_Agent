def build_prompt(task_prompt: str) -> list:
    return [
        {"role": "system", "content": (
            'Extract all named entities from the text. Return ONLY valid JSON with this exact structure: '
            '{"person": [...], "organization": [...], "location": [...], "date": [...]}. '
            "Use empty arrays for categories with no matches. Do not include any text outside the JSON object."
        )},
        {"role": "user", "content": task_prompt},
    ]
