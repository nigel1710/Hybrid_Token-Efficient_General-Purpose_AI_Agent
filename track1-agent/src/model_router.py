import re
import logging
from typing import List, Dict
from enum import Enum

logger = logging.getLogger(__name__)

# Tier assignment per category — edit this table after launch-day model IDs are known
CATEGORY_TIER = {
    "sentiment_classification": "CHEAP",
    "ner": "CHEAP",
    "factual_knowledge": "CHEAP",
    "summarisation": "MID",
    "code_debugging": "MID",
    "math_reasoning": "LARGE",
    "logical_reasoning": "LARGE",
    "code_generation": "MID",
}

# Manual override: map known model name substrings to a sort key (lower = cheaper)
# Edit once real model IDs are published on launch day
_SIZE_HINTS = [
    (r"405b", 405),
    (r"70b", 70),
    (r"34b", 34),
    (r"13b", 13),
    (r"8b", 8),
    (r"7b", 7),
    (r"3b", 3),
    (r"1b", 1),
]


def _model_size_key(model_id: str) -> int:
    lower = model_id.lower()
    for pattern, size in _SIZE_HINTS:
        if re.search(pattern, lower):
            return size
    return 999  # unknown → treat as large


def _build_tiers(allowed_models: List[str]) -> Dict[str, str]:
    sorted_models = sorted(allowed_models, key=_model_size_key)
    n = len(sorted_models)
    if n == 1:
        return {"CHEAP": sorted_models[0], "MID": sorted_models[0], "LARGE": sorted_models[0]}
    if n == 2:
        return {"CHEAP": sorted_models[0], "MID": sorted_models[1], "LARGE": sorted_models[1]}
    # 3+ models: cheap=first, large=last, mid=middle
    return {
        "CHEAP": sorted_models[0],
        "MID": sorted_models[n // 2],
        "LARGE": sorted_models[-1],
    }


_tiers: Dict[str, str] = {}
_allowed_set: set = set()


def init_router(allowed_models: List[str]):
    global _tiers, _allowed_set
    _allowed_set = set(allowed_models)
    _tiers = _build_tiers(allowed_models)
    logger.info("Model tiers: CHEAP=%s MID=%s LARGE=%s", _tiers["CHEAP"], _tiers["MID"], _tiers["LARGE"])

    # Validate all tier models are in allowed list
    for tier, model in _tiers.items():
        if model not in _allowed_set:
            logger.warning("Tier %s model %r not in ALLOWED_MODELS — falling back to first allowed model", tier, model)
            _tiers[tier] = allowed_models[0]


def get_model_for_category(category: str) -> str:
    tier = CATEGORY_TIER.get(category, "MID")
    model = _tiers.get(tier, list(_tiers.values())[0])
    logger.info("Category %r → tier %s → model %s", category, tier, model)
    return model
