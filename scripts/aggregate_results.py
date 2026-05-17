#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lqddn.metrics import flatten_results_row, per_label_rows, rows_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate LQDDN experiment results.")
    parser.add_argument("--root", required=True, help="Root output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summary_rows = []
    per_label = []
    ablations = []

    for metrics_path in root.rglob("metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        language = payload["language"]
        experiment = payload["experiment_name"]
        for split in ("validation", "test"):
            metrics = payload[split]
            summary_rows.append(flatten_results_row(language, experiment, split, metrics))
            per_label.extend(per_label_rows(language, experiment, split, metrics))
        ablations.append(
            {
                "language": language,
                "experiment": experiment,
                "macro_f1": payload["test"]["macro_f1"],
                "micro_f1": payload["test"]["micro_f1"],
                "weighted_f1": payload["test"]["weighted_f1"],
                "mAP": payload["test"]["mAP"],
                "hamming_loss": payload["test"]["hamming_loss"],
                "subset_accuracy": payload["test"]["subset_accuracy"],
            }
        )

    rows_to_csv(summary_rows, str(root / "results_summary.csv"))
    rows_to_csv(per_label, str(root / "per_label_results.csv"))
    rows_to_csv(ablations, str(root / "ablation_table.csv"))


if __name__ == "__main__":
    main()

