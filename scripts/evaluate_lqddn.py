#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lqddn.calibration import TemperatureBiasCalibrator
from lqddn.config import load_config
from lqddn.trainer import (
    apply_calibration,
    build_tokenizer,
    collect_outputs,
    compute_eval_losses,
    create_dataloaders,
    evaluate_arrays,
    format_metrics,
)
from lqddn.data import build_label_graph, compute_label_priors, compute_label_weights
from lqddn.metrics import tune_thresholds
from lqddn.model import LQDDNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved LQDDN checkpoint without retraining.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Path to trained best_model.pt.")
    parser.add_argument("--output-dir", help="Optional directory to save refreshed metrics.")
    return parser.parse_args()


def load_calibrator(path: Path) -> TemperatureBiasCalibrator | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    calibrator = TemperatureBiasCalibrator(len(payload["bias"]))
    calibrator.log_temperature.data.copy_(torch.tensor(payload["log_temperature"], dtype=torch.float32))
    calibrator.bias.data.copy_(torch.tensor(payload["bias"], dtype=torch.float32))
    return calibrator


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = build_tokenizer(config["model"]["backbone"])
    dataloaders, metadata = create_dataloaders(config, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_labels = metadata["train_labels"]
    label_weights = torch.tensor(compute_label_weights(train_labels), dtype=torch.float32, device=device)
    label_priors = torch.tensor(compute_label_priors(train_labels), dtype=torch.float32, device=device)
    adjacency = torch.tensor(build_label_graph(train_labels), dtype=torch.float32, device=device)

    model = LQDDNModel(
        backbone_name=config["model"]["backbone"],
        num_labels=len(metadata["label_names"]),
        num_attention_heads=config["model"]["num_attention_heads"],
        dropout=config["model"]["dropout"],
        use_label_queries=config["model"]["use_label_queries"],
        use_static_graph=config["model"]["use_static_graph"],
        use_dynamic_dependency=config["model"]["use_dynamic_dependency"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    validation_arrays = collect_outputs(model, dataloaders["validation"], device, adjacency, desc="Validation Eval")
    test_arrays = collect_outputs(model, dataloaders["test"], device, adjacency, desc="Test Eval")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model).resolve().parent
    calibrator = load_calibrator(output_dir / "calibrator.json")
    validation_probabilities = apply_calibration(validation_arrays["probabilities"], validation_arrays["logits"], calibrator)
    test_probabilities = apply_calibration(test_arrays["probabilities"], test_arrays["logits"], calibrator)

    thresholds = None
    if config["inference"]["threshold_mode"] == "tuned":
        thresholds = tune_thresholds(validation_arrays["targets"], validation_probabilities)

    validation_result = evaluate_arrays(
        validation_arrays["targets"],
        validation_arrays["logits"],
        validation_probabilities,
        metadata["label_names"],
        config["inference"]["threshold_mode"],
        config["inference"]["fixed_threshold"],
        thresholds,
    )
    test_result = evaluate_arrays(
        test_arrays["targets"],
        test_arrays["logits"],
        test_probabilities,
        metadata["label_names"],
        config["inference"]["threshold_mode"],
        config["inference"]["fixed_threshold"],
        thresholds,
    )
    validation_result["metrics"].update(compute_eval_losses(config, validation_arrays["logits"], validation_arrays["targets"], label_weights, label_priors))
    test_result["metrics"].update(compute_eval_losses(config, test_arrays["logits"], test_arrays["targets"], label_weights, label_priors))

    payload = {
        "language": config["language"]["name"],
        "experiment_name": output_dir.name,
        "validation": validation_result["metrics"],
        "test": test_result["metrics"],
    }
    print("Validation:", format_metrics(validation_result["metrics"]))
    print("Test:", format_metrics(test_result["metrics"]))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

