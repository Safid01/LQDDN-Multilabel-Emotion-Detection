from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, label_weights: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=label_weights)


def asymmetric_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    if clip > 0:
        probabilities = torch.clamp(probabilities, min=clip, max=1.0 - clip)
    pos_term = targets * torch.log(probabilities.clamp(min=epsilon)) * ((1.0 - probabilities) ** gamma_pos)
    neg_term = (1.0 - targets) * torch.log((1.0 - probabilities).clamp(min=epsilon)) * (probabilities ** gamma_neg)
    return -(pos_term + neg_term).mean()


def multilabel_ranking_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    total = logits.new_tensor(0.0)
    count = 0
    for sample_logits, sample_pos, sample_neg in zip(logits, pos_mask, neg_mask, strict=False):
        pos_scores = sample_logits[sample_pos]
        neg_scores = sample_logits[sample_neg]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pairwise = margin - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)
        total = total + F.relu(pairwise).mean()
        count += 1
    if count == 0:
        return logits.new_tensor(0.0)
    return total / count


def rare_label_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
    gamma: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    focal_factor = torch.where(targets > 0.5, (1.0 - probabilities) ** gamma, probabilities ** gamma)
    bce = -(
        targets * torch.log(probabilities.clamp(min=epsilon))
        + (1.0 - targets) * torch.log((1.0 - probabilities).clamp(min=epsilon))
    )
    scaled_weights = label_weights / label_weights.mean().clamp(min=epsilon)
    return (focal_factor * bce * scaled_weights.unsqueeze(0)).mean()

