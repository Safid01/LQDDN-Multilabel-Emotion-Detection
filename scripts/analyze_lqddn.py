#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lqddn.analysis import (
    ensure_dir,
    labels_from_binary,
    plot_gate_bars,
    plot_heatmap,
    save_json,
    save_rows_csv,
    top_tokens_for_label,
    torch_to_numpy,
)
from lqddn.config import load_config
from lqddn.data import build_label_graph, dataframe_to_arrays, load_dataset_bundle
from lqddn.trainer import build_tokenizer, create_dataloaders, move_batch_to_device
from lqddn.metrics import tune_thresholds
from lqddn.model import LQDDNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interpretability and error analysis artifacts for LQDDN.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Path to trained best_model.pt.")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Split to analyze.")
    parser.add_argument("--output-dir", required=True, help="Directory to save analysis outputs.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of qualitative samples to analyze.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis.")
    return parser.parse_args()


def build_model(config: dict, num_labels: int, device: torch.device) -> LQDDNModel:
    model = LQDDNModel(
        backbone_name=config["model"]["backbone"],
        num_labels=num_labels,
        num_attention_heads=config["model"]["num_attention_heads"],
        dropout=config["model"]["dropout"],
        use_label_queries=config["model"]["use_label_queries"],
        use_static_graph=config["model"]["use_static_graph"],
        use_dynamic_dependency=config["model"]["use_dynamic_dependency"],
    ).to(device)
    return model


def decode_tokens(tokenizer, input_ids: torch.Tensor) -> list[str]:
    return tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = build_tokenizer(config["model"]["backbone"])
    dataloaders, metadata = create_dataloaders(config, tokenizer)
    bundle = load_dataset_bundle(config["data"]["splits"])

    split_df = bundle.validation_df if args.split == "validation" else bundle.test_df
    split_texts, split_labels = dataframe_to_arrays(split_df, bundle.text_column, bundle.label_columns, bundle.label_format)
    label_names = metadata["label_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, len(label_names), device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    adjacency = torch.tensor(build_label_graph(metadata["train_labels"]), dtype=torch.float32, device=device)

    output_root = ensure_dir(args.output_dir)
    heatmap_dir = ensure_dir(output_root / "heatmaps")

    indices = list(range(min(args.num_samples, len(split_texts))))
    dataset = dataloaders[args.split].dataset
    subset_loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)

    case_studies = []
    error_rows = []
    cross_attention_all = []
    dynamic_attention_all = []
    gate_all = []
    logits_all = []
    targets_all = []

    with torch.no_grad():
        cursor = 0
        for batch in subset_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, adjacency, return_analysis=True)
            probabilities = outputs.probabilities.detach().cpu().numpy()
            logits = outputs.logits.detach().cpu().numpy()
            targets = batch["labels"].detach().cpu().numpy()
            cross_weights = torch_to_numpy(outputs.cross_attention_weights)
            dynamic_weights = torch_to_numpy(outputs.dynamic_attention_weights)
            gates = torch_to_numpy(outputs.fusion_gates)
            input_ids = batch["input_ids"].detach().cpu()
            attention_mask = batch["attention_mask"].detach().cpu() if "attention_mask" in batch else None

            logits_all.append(logits)
            targets_all.append(targets)
            if cross_weights is not None:
                cross_attention_all.append(cross_weights)
            if dynamic_weights is not None:
                dynamic_attention_all.append(dynamic_weights)
            if gates is not None:
                gate_all.append(gates)

            for row_idx in range(targets.shape[0]):
                sample_idx = indices[cursor]
                text = split_texts[sample_idx]
                sample_targets = targets[row_idx]
                sample_probs = probabilities[row_idx]
                sample_preds = (sample_probs > 0.5).astype(np.float32)
                tokens = decode_tokens(tokenizer, input_ids[row_idx])
                if attention_mask is not None:
                    valid_len = int(attention_mask[row_idx].sum().item())
                    tokens = tokens[:valid_len]
                else:
                    valid_len = len(tokens)

                sample_case = {
                    "index": sample_idx,
                    "text": text,
                    "gold_labels": labels_from_binary(sample_targets, label_names),
                    "predicted_labels": labels_from_binary(sample_preds, label_names),
                    "top_probabilities": [
                        {
                            "label": label_names[idx],
                            "probability": float(sample_probs[idx]),
                            "gold": int(sample_targets[idx]),
                            "predicted": int(sample_preds[idx]),
                        }
                        for idx in np.argsort(sample_probs)[::-1][: min(5, len(label_names))]
                    ],
                }

                if cross_weights is not None:
                    sample_cross = cross_weights[row_idx, :, :valid_len]
                    top_tokens = {
                        label_names[label_idx]: top_tokens_for_label(tokens, sample_cross[label_idx], top_k=5)
                        for label_idx in range(len(label_names))
                    }
                    sample_case["top_attention_tokens"] = top_tokens
                    plot_heatmap(
                        sample_cross,
                        tokens,
                        label_names,
                        f"Cross-Attention Sample {sample_idx}",
                        heatmap_dir / f"sample_{sample_idx}_cross_attention.png",
                        cmap="Purples",
                    )
                if dynamic_weights is not None:
                    sample_dynamic = dynamic_weights[row_idx]
                    plot_heatmap(
                        sample_dynamic,
                        label_names,
                        label_names,
                        f"Dynamic Label Interaction Sample {sample_idx}",
                        heatmap_dir / f"sample_{sample_idx}_dynamic_attention.png",
                        cmap="Blues",
                    )
                if gates is not None:
                    sample_case["fusion_gates"] = {
                        label_names[label_idx]: {
                            "static": float(gates[row_idx, label_idx, 0]),
                            "dynamic": float(gates[row_idx, label_idx, 1]),
                        }
                        for label_idx in range(len(label_names))
                    }

                false_positives = [label_names[idx] for idx in range(len(label_names)) if sample_preds[idx] > 0.5 and sample_targets[idx] < 0.5]
                false_negatives = [label_names[idx] for idx in range(len(label_names)) if sample_preds[idx] < 0.5 and sample_targets[idx] > 0.5]
                if false_positives or false_negatives:
                    error_rows.append(
                        {
                            "index": sample_idx,
                            "split": args.split,
                            "text": text,
                            "gold_labels": ", ".join(sample_case["gold_labels"]),
                            "predicted_labels": ", ".join(sample_case["predicted_labels"]),
                            "false_positives": ", ".join(false_positives),
                            "false_negatives": ", ".join(false_negatives),
                        }
                    )

                case_studies.append(sample_case)
                cursor += 1

    logits_all_np = np.concatenate(logits_all, axis=0)
    targets_all_np = np.concatenate(targets_all, axis=0)
    threshold_values = tune_thresholds(targets_all_np, 1.0 / (1.0 + np.exp(-logits_all_np)))

    analysis_summary: dict[str, object] = {
        "split": args.split,
        "num_samples": len(case_studies),
        "thresholds_from_subset": {label: float(threshold_values[idx]) for idx, label in enumerate(label_names)},
    }

    if cross_attention_all:
        cross_mean = np.concatenate(cross_attention_all, axis=0).mean(axis=0)
        analysis_summary["average_cross_attention_shape"] = list(cross_mean.shape)
    if dynamic_attention_all:
        dynamic_mean = np.concatenate(dynamic_attention_all, axis=0).mean(axis=0)
        plot_heatmap(
            dynamic_mean,
            label_names,
            label_names,
            f"Average Dynamic Label Interaction ({args.split})",
            heatmap_dir / f"{args.split}_dynamic_attention_average.png",
            cmap="Blues",
        )
        analysis_summary["average_dynamic_attention"] = dynamic_mean.tolist()
    if gate_all:
        gate_mean = np.concatenate(gate_all, axis=0).mean(axis=0)
        plot_gate_bars(gate_mean, label_names, heatmap_dir / f"{args.split}_fusion_gates_average.png")
        analysis_summary["average_fusion_gates"] = {
            label_names[idx]: {"static": float(gate_mean[idx, 0]), "dynamic": float(gate_mean[idx, 1])}
            for idx in range(len(label_names))
        }

    adjacency = adjacency.detach().cpu().numpy()
    plot_heatmap(
        adjacency,
        label_names,
        label_names,
        "Static Label Co-occurrence Graph",
        heatmap_dir / "static_label_graph.png",
        cmap="Greens",
    )

    save_json(analysis_summary, output_root / "analysis_summary.json")
    save_json({"cases": case_studies}, output_root / "case_studies.json")
    save_rows_csv(error_rows, output_root / "error_analysis.csv")
    print(f"Saved analysis artifacts to {output_root}")


if __name__ == "__main__":
    main()
