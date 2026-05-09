"""Utility helpers: configuration loading, logging setup, path helpers, RNG seeding."""
from __future__ import annotations

import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_SAVED = PROJECT_ROOT / "models" / "saved"
SIM_RESULTS = PROJECT_ROOT / "simulation" / "results"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_SAVED, SIM_RESULTS):
    _p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean console format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    """Seed both numpy and the python random module for determinism."""
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Team-name normalisation. The international_results dataset and our hardcoded
# squad-value/xGD dictionaries occasionally disagree (e.g. "Czech Republic" vs
# "Czechia"). Normalise everything to a single canonical form.
# ---------------------------------------------------------------------------
TEAM_ALIASES: dict[str, str] = {
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "United States": "USA",
    "United States of America": "USA",
    "USA Men's National Team": "USA",
    "Curaçao": "Curacao",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
}


def normalise_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip(), name.strip())
