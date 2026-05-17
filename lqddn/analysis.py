from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    output_path: str | Path,
    cmap: str = "viridis",
) -> None:
    output_path = Path(output_path)
    plt.figure(figsize=(max(6, len(x_labels) * 0.5), max(4, len(y_labels) * 0.45)))
    plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_gate_bars(gates: np.ndarray, label_names: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    static_gate = gates[:, 0]
    dynamic_gate = gates[:, 1]
    x = np.arange(len(label_names))
    width = 0.38
    plt.figure(figsize=(max(8, len(label_names) * 0.6), 4.8))
    plt.bar(x - width / 2, static_gate, width=width, label="Static Gate")
    plt.bar(x + width / 2, dynamic_gate, width=width, label="Dynamic Gate")
    plt.xticks(x, label_names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average Gate Weight")
    plt.title("Average Fusion Gate Weights by Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def labels_from_binary(row: np.ndarray, label_names: list[str]) -> list[str]:
    return [label for label, flag in zip(label_names, row.tolist(), strict=False) if flag >= 0.5]


def top_tokens_for_label(tokens: list[str], weights: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    limit = min(top_k, len(tokens))
    indices = np.argsort(weights)[::-1][:limit]
    return [{"token": tokens[idx], "weight": float(weights[idx])} for idx in indices]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def torch_to_numpy(tensor: torch.Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()

