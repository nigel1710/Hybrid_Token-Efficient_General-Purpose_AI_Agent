import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real model list — published for Track 1 on launch day.
# Retune tier assignments here after running tasks_all_categories.json against
# real Fireworks calls and observing accuracy per category.
# ---------------------------------------------------------------------------

# Explicit tier assignment for each known model ID.
# NOTE: The old approach used fuzzy size-hint parsing (e.g. "8b", "70b" substrings)
# to auto-sort unknown model names into tiers. That heuristic is no longer needed
# now that the real list is fixed — but the fallback logic below still handles the
# case where the harness injects a different ALLOWED_MODELS list at runtime.
_MODEL_TIERS: Dict[str, str] = {
    # MoE with small active params — fastest/cheapest per token
    "accounts/fireworks/models/gemma-4-26b-a4b-it":   "CHEAP",
    # Quantized 31b — near-CHEAP cost with more accuracy than the a4b variant
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4": "CHEAP",
    # Full 31b — solid mid-tier general model
    "accounts/fireworks/models/gemma-4-31b-it":        "MID",
    # Code-specialized model — used as a fixed override for code categories (see below)
    "accounts/fireworks/models/kimi-k2p7-code":        "MID",
    # Strongest general reasoning — reserved for math and logic
    "accounts/fireworks/models/minimax-m3":            "LARGE",
}

# Fixed routing override: always use this model for code categories,
# regardless of tier, because it is explicitly code-specialized.
# Validated at startup; falls back to MID tier if absent from ALLOWED_MODELS.
_CODE_MODEL = "accounts/fireworks/models/kimi-k2p7-code"
_CODE_CATEGORIES = {"code_debugging", "code_generation"}

# Category → tier mapping.
# code_debugging and code_generation are handled via _CODE_MODEL override above.
CATEGORY_TIER: Dict[str, str] = {
    "sentiment_classification": "CHEAP",
    "ner":                       "CHEAP",
    "factual_knowledge":         "MID",
    "summarisation":             "MID",
    "math_reasoning":            "LARGE",
    "logical_reasoning":         "LARGE",
    # code categories below are overridden by _CODE_MODEL — tier is a fallback only
    "code_debugging":            "MID",
    "code_generation":           "MID",
}
# ---------------------------------------------------------------------------

_tiers: Dict[str, str] = {}
_allowed_set: set = set()
_code_model_available: bool = False
_unavailable: set = set()  # models that failed warm-up for this run


def _build_tiers(allowed_models: List[str]) -> Dict[str, str]:
    """Map CHEAP/MID/LARGE tier names to actual model IDs from the allowed list."""
    # Prefer the explicit tier table; fall back to positional assignment for
    # any model IDs not present in _MODEL_TIERS (harness may inject a different list).
    tier_to_model: Dict[str, Optional[str]] = {"CHEAP": None, "MID": None, "LARGE": None}

    for model in allowed_models:
        tier = _MODEL_TIERS.get(model)
        if tier and tier_to_model[tier] is None:
            tier_to_model[tier] = model

    # Fill any tier that got no match using positional fallback (first=CHEAP, last=LARGE)
    n = len(allowed_models)
    fallback = {
        "CHEAP": allowed_models[0],
        "MID":   allowed_models[n // 2],
        "LARGE": allowed_models[-1],
    }
    for tier in ("CHEAP", "MID", "LARGE"):
        if tier_to_model[tier] is None:
            tier_to_model[tier] = fallback[tier]
            logger.warning("No model mapped to tier %s from known list — using positional fallback: %s",
                           tier, fallback[tier])

    return {k: v for k, v in tier_to_model.items()}


def init_router(allowed_models: List[str]):
    global _tiers, _allowed_set, _code_model_available
    _allowed_set = set(allowed_models)

    _tiers = _build_tiers(allowed_models)
    logger.info("Model tiers: CHEAP=%s  MID=%s  LARGE=%s",
                _tiers["CHEAP"], _tiers["MID"], _tiers["LARGE"])

    # Validate every tier model is actually in ALLOWED_MODELS
    for tier, model in _tiers.items():
        if model not in _allowed_set:
            logger.warning("Tier %s model %r not in ALLOWED_MODELS — falling back to first allowed model", tier, model)
            _tiers[tier] = allowed_models[0]

    # Validate the code-specialist override
    # Match by full ID or short suffix (handles both harness-injected and local .env formats)
    _code_model_available = any(_CODE_MODEL == m or m.endswith("/" + _CODE_MODEL.split("/")[-1]) for m in _allowed_set)
    if not _code_model_available:
        logger.warning("%r not in ALLOWED_MODELS — code categories will fall back to MID tier", _CODE_MODEL)
    else:
        logger.info("Code override model available: %s", _CODE_MODEL)


def mark_model_unavailable(models: set):
    """Called after warm-up (or mid-run) to exclude models that failed to deploy."""
    global _unavailable
    _unavailable |= models
    # Remap every tier whose assigned model is now unavailable
    all_models = list(_MODEL_TIERS.keys())
    for tier, assigned in list(_tiers.items()):
        if assigned in _unavailable:
            # Pick the first known model not unavailable; last resort: keep current
            fallback = next(
                (m for m in all_models if m not in _unavailable and m in _allowed_set),
                assigned,
            )
            logger.warning(
                "Model %s unavailable — remapping tier %s to %s",
                assigned, tier, fallback,
            )
            _tiers[tier] = fallback


def get_model_for_category(category: str) -> str:
    # Fixed override for code categories
    if category in _CODE_CATEGORIES and _code_model_available:
        code_model_id = next(
            (m for m in _allowed_set if m == _CODE_MODEL or m.endswith("/" + _CODE_MODEL.split("/")[-1])),
            None
        )
        if code_model_id and code_model_id not in _unavailable:
            logger.info("Category %r -> code override -> %s", category, code_model_id)
            return code_model_id

    tier = CATEGORY_TIER.get(category, "MID")
    model = _tiers.get(tier, list(_tiers.values())[0])
    logger.info("Category %r -> tier %s -> model %s", category, tier, model)
    return model
