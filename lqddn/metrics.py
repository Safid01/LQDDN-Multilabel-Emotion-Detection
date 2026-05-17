from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


def mean_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    for idx in range(targets.shape[1]):
        if len(np.unique(targets[:, idx])) < 2:
            continue
        scores.append(average_precision_score(targets[:, idx], probabilities[:, idx]))
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    metrics = {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "subset_accuracy": float(accuracy_score(targets, predictions)),
        "mAP": mean_average_precision(targets, probabilities),
        "per_label": [],
        "label_frequency": [],
        "predicted_label_frequency": [],
    }
    for idx, label in enumerate(label_names):
        metrics["per_label"].append(
            {
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
        metrics["label_frequency"].append(
            {
                "label": label,
                "count": int(targets[:, idx].sum()),
                "ratio": float(targets[:, idx].mean()),
            }
        )
        metrics["predicted_label_frequency"].append(
            {
                "label": label,
                "count": int(predictions[:, idx].sum()),
                "ratio": float(predictions[:, idx].mean()),
            }
        )
    return metrics


def tune_thresholds(targets: np.ndarray, probabilities: np.ndarray, search_space: np.ndarray | None = None) -> list[float]:
    if search_space is None:
        search_space = np.arange(0.1, 0.91, 0.05)
    thresholds = []
    for idx in range(targets.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in search_space:
            predictions = (probabilities[:, idx] > threshold).astype(int)
            score = f1_score(targets[:, idx], predictions, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def flatten_results_row(language: str, experiment: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": language,
        "experiment": experiment,
        "split": split,
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "hamming_loss": metrics["hamming_loss"],
        "subset_accuracy": metrics["subset_accuracy"],
        "mAP": metrics["mAP"],
    }


def per_label_rows(language: str, experiment: str, split: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in metrics["per_label"]:
        row = {"language": language, "experiment": experiment, "split": split}
        row.update(item)
        rows.append(row)
    return rows


def rows_to_csv(rows: list[dict[str, Any]], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)

