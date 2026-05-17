from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ABLATION_PRESETS = {
    "A5_full_lqddn": {
        "model": {
            "use_label_queries": True,
            "use_static_graph": True,
            "use_dynamic_dependency": True,
        },
        "calibration": {
            "enabled": True,
        },
        "inference": {
            "threshold_mode": "tuned",
        },
    },
    "A6_full_no_graph": {
        "model": {
            "use_label_queries": True,
            "use_static_graph": False,
            "use_dynamic_dependency": True,
        },
        "calibration": {
            "enabled": True,
        },
        "inference": {
            "threshold_mode": "tuned",
        },
    },
    "A7_full_no_dynamic": {
        "model": {
            "use_label_queries": True,
            "use_static_graph": True,
            "use_dynamic_dependency": False,
        },
        "calibration": {
            "enabled": True,
        },
        "inference": {
            "threshold_mode": "tuned",
        },
    },
    "A8_full_no_ranking": {
        "model": {
            "use_label_queries": True,
            "use_static_graph": True,
            "use_dynamic_dependency": True,
        },
        "loss": {
            "lambda_ranking": 0.0,
        },
        "calibration": {
            "enabled": True,
        },
        "inference": {
            "threshold_mode": "tuned",
        },
    },
    "A9_full_no_calibration": {
        "model": {
            "use_label_queries": True,
            "use_static_graph": True,
            "use_dynamic_dependency": True,
        },
        "loss": {
            "lambda_calibration": 0.0,
        },
        "calibration": {
            "enabled": False,
        },
        "inference": {
            "threshold_mode": "tuned",
        },
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, ablation: str | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(config_path.resolve())
    if ablation:
        if ablation not in ABLATION_PRESETS:
            raise ValueError(f"Unknown ablation: {ablation}")
        config = deep_update(config, ABLATION_PRESETS[ablation])
        config["experiment_name"] = ablation
    else:
        config.setdefault("experiment_name", "default")
    return config
