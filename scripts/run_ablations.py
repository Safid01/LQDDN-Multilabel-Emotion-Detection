#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lqddn.config import ABLATION_PRESETS
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all LQDDN ablations for one language config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language = "english" if "english" in Path(args.config).stem.lower() else "chinese"
    ablations = sorted(ABLATION_PRESETS.keys())
    for ablation in tqdm(ablations, desc=f"{language} ablations"):
        command = [
            args.python,
            "scripts/train_lqddn.py",
            "--config",
            args.config,
            "--ablation",
            ablation,
            "--output-dir",
            str(Path(args.output_root) / language / ablation),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
