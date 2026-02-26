"""Minimal YAML config loader for experiments."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file; return nested dict. Path can be str or Path."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    logger.debug("Loaded config from %s", path)
    return cfg


def get_nested(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested value by dot-separated key (e.g. 'train.epochs')."""
    keys = key_path.split(".")
    current: Any = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current
