from pathlib import Path

import yaml


DEFAULT_SYNTHETIC_IV_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "synthetic_iv.yaml"


def load_synthetic_iv_config(path=None):
    """Load the synthetic IV generation config."""
    config_path = Path(path) if path is not None else DEFAULT_SYNTHETIC_IV_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(cfg=None):
    """Accept either an already-loaded config dict or load the default config."""
    return load_synthetic_iv_config() if cfg is None else cfg


def config_value(value):
    """Resolve scalar config values, using range midpoints for min/max dicts."""
    if isinstance(value, dict) and "min" in value and "max" in value:
        return 0.5 * (value["min"] + value["max"])
    return value
