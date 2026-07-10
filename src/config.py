import os
import sys
import logging
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Config:
    api_key: str
    base_url: str
    allowed_models: List[str]


def _load_env():
    if os.environ.get("LOCAL_DEV", "").lower() == "true":
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded .env for local dev")
        except ImportError:
            pass


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _validate_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def load_config() -> Config:
    _load_env()

    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip()
    allowed_models_raw = os.environ.get("ALLOWED_MODELS", "").strip()

    errors = []
    if not api_key:
        errors.append("FIREWORKS_API_KEY is missing or empty")
    if not base_url:
        errors.append("FIREWORKS_BASE_URL is missing or empty")
    elif not _validate_url(base_url):
        errors.append(f"FIREWORKS_BASE_URL is not a valid URL: {base_url!r}")
    if not allowed_models_raw:
        errors.append("ALLOWED_MODELS is missing or empty")

    allowed_models = [m.strip() for m in allowed_models_raw.split(",") if m.strip()]
    if not errors and not allowed_models:
        errors.append("ALLOWED_MODELS contains no valid model IDs after parsing")

    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        sys.exit(1)

    return Config(
        api_key=api_key,
        base_url=_normalize_base_url(base_url),
        allowed_models=allowed_models,
    )
