"""Load YAML config files with caching."""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load a YAML config file by name (without .yaml extension)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def engine() -> dict:
    return load("engine")


def eval_() -> dict:
    return load("eval")


def training() -> dict:
    return load("training")


def deploy() -> dict:
    return load("deploy")
