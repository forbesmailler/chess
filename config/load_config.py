"""Load YAML config files with caching."""

import copy
import platform
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
    cfg = copy.deepcopy(load("deploy"))
    exe = Path(cfg["paths"]["bot_exe"])
    if platform.system() == "Windows":
        exe = exe.parent / "Release" / (exe.name + ".exe")
    cfg["paths"]["bot_exe"] = str(exe)
    return cfg
