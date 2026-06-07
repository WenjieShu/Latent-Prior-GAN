"""Configuration helpers for training entry points."""

import json
import os

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime for --config users.
    yaml = None


def load_config(path):
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required when --config is used.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_config(args):
    os.makedirs(args.yaml_dir, exist_ok=True)
    path = os.path.join(args.yaml_dir, f"{args.saver_name_pre}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    return path
