import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
load_dotenv(override=True)


@dataclass
class Config:
    api_key: str
    base_url: str
    allowed_models: List[str]


def load_config() -> Config:
    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip()

    # Comma-separated list of permitted Fireworks AI model IDs — used as-is
    allowed_models = [
        model.strip()
        for model in os.environ.get("ALLOWED_MODELS", "").split(",")
        if model.strip()
    ]

    return Config(
        api_key=api_key,
        base_url=base_url,
        allowed_models=allowed_models,
    )
