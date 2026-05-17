from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TEXT_CANDIDATES = {"text", "tweet", "sentence", "content", "utterance", "review"}
ID_CANDIDATES = {"id", "uid", "guid", "index"}
POSITIVE_LABEL_VALUES = {"1", "1.0", "true", "yes", "y", "t"}
NEGATIVE_LABEL_VALUES = {"0", "0.0", "false", "no", "n", "f"}


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    text_column: str
    label_columns: list[str]
    label_format: str


def read_split(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def _looks_binary(series: pd.Series) -> bool:
    normalized = {
        str(value).strip().lower()
        for value in series.dropna().tolist()
        if str(value).strip() != ""
    }
    return normalized.issubset(POSITIVE_LABEL_VALUES | NEGATIVE_LABEL_VALUES)


def _find_text_column(df: pd.DataFrame) -> str:
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in TEXT_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    text_like = []
    for column in df.columns:
        if column.lower() in ID_CANDIDATES:
            continue
        non_null = df[column].dropna()
        if non_null.empty:
            continue
        avg_len = non_null.astype(str).map(len).mean()
        unique_ratio = non_null.nunique() / max(len(non_null), 1)
        if avg_len > 12 and unique_ratio > 0.5:
            text_like.append((avg_len, column))
    if not text_like:
        raise ValueError("Could not infer text column.")
    text_like.sort(reverse=True)
    return text_like[0][1]


def _parse_label_string(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[\s,;|/]+", text) if token.strip()]


def infer_schema(df: pd.DataFrame) -> tuple[str, list[str], str]:
    text_column = _find_text_column(df)
    binary_columns = [
        column
        for column in df.columns
        if column != text_column and column.lower() not in ID_CANDIDATES and _looks_binary(df[column])
    ]
    if binary_columns:
        return text_column, binary_columns, "binary_columns"

    for column in (column for column in df.columns if column != text_column):
        labels = df[column].map(_parse_label_string)
        if labels.map(len).sum() > 0:
            vocab = sorted({label for row in labels.tolist() for label in row})
            if vocab:
                return text_column, vocab, f"label_string:{column}"
    raise ValueError("Could not infer multilabel schema.")


def _build_binary_matrix(df: pd.DataFrame, label_columns: list[str]) -> np.ndarray:
    matrix = df[label_columns].copy()
    for column in label_columns:
        matrix[column] = (
            matrix[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda value: 1.0 if value in POSITIVE_LABEL_VALUES else 0.0)
        )
    return matrix.to_numpy(dtype=np.float32)


def _build_string_matrix(df: pd.DataFrame, label_vocab: list[str], label_column: str) -> np.ndarray:
    labels = df[label_column].map(_parse_label_string).tolist()
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    matrix = np.zeros((len(labels), len(label_vocab)), dtype=np.float32)
    for row_idx, row_labels in enumerate(labels):
        for label in row_labels:
            if label in label_to_idx:
                matrix[row_idx, label_to_idx[label]] = 1.0
    return matrix


def dataframe_to_arrays(
    df: pd.DataFrame,
    text_column: str,
    label_columns: list[str],
    label_format: str,
) -> tuple[list[str], np.ndarray]:
    texts = df[text_column].fillna("").astype(str).tolist()
    if label_format == "binary_columns":
        labels = _build_binary_matrix(df, label_columns)
    else:
        labels = _build_string_matrix(df, label_columns, label_format.split(":", maxsplit=1)[1])
    return texts, labels


def load_dataset_bundle(split_paths: dict[str, str]) -> DatasetBundle:
    train_df = read_split(split_paths["train"])
    validation_df = read_split(split_paths["validation"])
    test_df = read_split(split_paths["test"])
    text_column, label_columns, label_format = infer_schema(train_df)
    return DatasetBundle(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        text_column=text_column,
        label_columns=label_columns,
        label_format=label_format,
    )


class MultiLabelEmotionDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


def compute_label_weights(labels: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    num_examples = labels.shape[0]
    positive_counts = labels.sum(axis=0)
    return ((num_examples - positive_counts) / (positive_counts + epsilon)).astype(np.float32)


def compute_label_priors(labels: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    priors = labels.mean(axis=0)
    return np.clip(priors, epsilon, 1.0 - epsilon).astype(np.float32)


def build_label_graph(labels: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    co_occurrence = labels.T @ labels
    co_occurrence = co_occurrence + np.eye(co_occurrence.shape[0], dtype=np.float32)
    row_sums = co_occurrence.sum(axis=1, keepdims=True) + epsilon
    return (co_occurrence / row_sums).astype(np.float32)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

