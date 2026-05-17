#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lqddn.config import ABLATION_PRESETS, load_config
from lqddn.trainer import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LQDDN for monolingual multilabel emotion recognition.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--ablation", choices=sorted(ABLATION_PRESETS.keys()), help="Optional ablation preset.")
    parser.add_argument("--output-dir", help="Optional explicit output directory.")
    parser.add_argument("--backbone", help="Optional backbone override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, ablation=args.ablation)
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    language = config["language"]["name"].lower()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / language / config["experiment_name"]
    train_and_evaluate(config, output_dir)


if __name__ == "__main__":
    main()

