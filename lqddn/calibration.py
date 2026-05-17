from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureBiasCalibrator(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_labels))
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature).unsqueeze(0)
        return logits / temperature + self.bias.unsqueeze(0)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(logits))


def fit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    learning_rate: float = 1.0,
) -> TemperatureBiasCalibrator:
    device = torch.device("cpu")
    model = TemperatureBiasCalibrator(logits.shape[1]).to(device)
    logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        calibrated = model(logits_tensor)
        loss = F.binary_cross_entropy_with_logits(calibrated, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def save_calibrator(calibrator: TemperatureBiasCalibrator, path: str | Path) -> None:
    payload = {
        "log_temperature": calibrator.log_temperature.detach().cpu().tolist(),
        "bias": calibrator.bias.detach().cpu().tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

