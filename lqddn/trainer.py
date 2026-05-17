from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lqddn.calibration import fit_calibrator, save_calibrator
from lqddn.data import (
    MultiLabelEmotionDataset,
    build_label_graph,
    compute_label_priors,
    compute_label_weights,
    dataframe_to_arrays,
    load_dataset_bundle,
    save_json,
)
from lqddn.hf_utils import disable_hf_safetensors_auto_conversion
from lqddn.losses import (
    asymmetric_bce_loss,
    multilabel_ranking_loss,
    rare_label_focal_loss,
    weighted_bce_loss,
)
from lqddn.metrics import compute_metrics, tune_thresholds
from lqddn.model import LQDDNModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(backbone_name: str):
    disable_hf_safetensors_auto_conversion()
    return AutoTokenizer.from_pretrained(backbone_name, use_fast=True)


def create_dataloaders(config: dict[str, Any], tokenizer):
    bundle = load_dataset_bundle(config["data"]["splits"])
    train_texts, train_labels = dataframe_to_arrays(bundle.train_df, bundle.text_column, bundle.label_columns, bundle.label_format)
    val_texts, val_labels = dataframe_to_arrays(bundle.validation_df, bundle.text_column, bundle.label_columns, bundle.label_format)
    test_texts, test_labels = dataframe_to_arrays(bundle.test_df, bundle.text_column, bundle.label_columns, bundle.label_format)

    dataset_kwargs = {"tokenizer": tokenizer, "max_length": config["data"]["max_length"]}
    train_dataset = MultiLabelEmotionDataset(train_texts, train_labels, **dataset_kwargs)
    validation_dataset = MultiLabelEmotionDataset(val_texts, val_labels, **dataset_kwargs)
    test_dataset = MultiLabelEmotionDataset(test_texts, test_labels, **dataset_kwargs)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "validation": DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    metadata = {
        "label_names": bundle.label_columns,
        "text_column": bundle.text_column,
        "label_format": bundle.label_format,
        "train_labels": train_labels,
        "validation_labels": val_labels,
        "test_labels": test_labels,
    }
    return dataloaders, metadata


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def collect_outputs(
    model: LQDDNModel,
    dataloader: DataLoader,
    device: torch.device,
    adjacency: torch.Tensor,
    desc: str,
) -> dict[str, np.ndarray]:
    model.eval()
    all_targets = []
    all_logits = []
    all_probabilities = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, adjacency)
            all_targets.append(batch["labels"].cpu().numpy())
            all_logits.append(outputs.logits.cpu().numpy())
            all_probabilities.append(outputs.probabilities.cpu().numpy())
    return {
        "targets": np.concatenate(all_targets, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "probabilities": np.concatenate(all_probabilities, axis=0),
    }


def apply_calibration(probabilities: np.ndarray, logits: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return probabilities
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    calibrated = calibrator.predict_proba(logits_tensor).detach().cpu().numpy()
    return calibrated


def evaluate_arrays(
    targets: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    threshold_mode: str,
    fixed_threshold: float,
    thresholds: list[float] | None,
) -> dict[str, Any]:
    if threshold_mode == "tuned" and thresholds is not None:
        threshold_array = np.asarray(thresholds, dtype=np.float32)[None, :]
        predictions = (probabilities > threshold_array).astype(np.float32)
    else:
        predictions = (probabilities > fixed_threshold).astype(np.float32)
    metrics = compute_metrics(targets, probabilities, predictions, label_names)
    metrics["num_examples"] = int(targets.shape[0])
    return {
        "metrics": metrics,
        "targets": targets,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def compute_eval_losses(
    config: dict[str, Any],
    logits: np.ndarray,
    targets: np.ndarray,
    label_weights: torch.Tensor,
    label_priors: torch.Tensor,
) -> dict[str, float]:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    cpu_label_weights = label_weights.detach().cpu()
    cpu_label_priors = label_priors.detach().cpu()

    classification_loss = build_classification_loss(config, logits_tensor, targets_tensor, cpu_label_weights)
    ranking_loss = multilabel_ranking_loss(logits_tensor, targets_tensor, margin=config["loss"]["ranking_margin"])
    calibration_loss = rare_label_focal_loss(
        logits_tensor,
        targets_tensor,
        label_weights=cpu_label_priors.reciprocal(),
        gamma=config["loss"]["calibration_gamma"],
    )
    total_loss = (
        classification_loss
        + config["loss"]["lambda_ranking"] * ranking_loss
        + config["loss"]["lambda_calibration"] * calibration_loss
    )
    return {
        "classification_loss": float(classification_loss.item()),
        "ranking_loss": float(ranking_loss.item()),
        "calibration_loss": float(calibration_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def build_classification_loss(
    config: dict[str, Any],
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
) -> torch.Tensor:
    loss_type = config["loss"]["classification"]
    if loss_type == "asymmetric":
        return asymmetric_bce_loss(
            logits,
            targets,
            gamma_neg=config["loss"]["asymmetric_gamma_neg"],
            gamma_pos=config["loss"]["asymmetric_gamma_pos"],
            clip=config["loss"]["asymmetric_clip"],
        )
    return weighted_bce_loss(logits, targets, label_weights)


def select_metric(metrics: dict[str, Any], monitor_metric: str) -> float:
    if monitor_metric not in metrics:
        raise ValueError(f"Unsupported monitor metric: {monitor_metric}")
    return float(metrics[monitor_metric])


def format_metrics(metrics: dict[str, Any]) -> str:
    base = (
        f"micro_f1={metrics['micro_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"hamming_loss={metrics['hamming_loss']:.4f} "
        f"subset_accuracy={metrics['subset_accuracy']:.4f} "
        f"mAP={metrics['mAP']:.4f}"
    )
    if "ranking_loss" in metrics:
        base += (
            f" classification_loss={metrics['classification_loss']:.4f}"
            f" ranking_loss={metrics['ranking_loss']:.4f}"
            f" calibration_loss={metrics['calibration_loss']:.4f}"
            f" total_loss={metrics['total_loss']:.4f}"
        )
    return base


def train_and_evaluate(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    set_seed(config["training"]["seed"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(config["model"]["backbone"])
    tokenizer.save_pretrained(output_dir / "tokenizer")

    dataloaders, metadata = create_dataloaders(config, tokenizer)
    label_names = metadata["label_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = metadata["train_labels"]
    label_weights = torch.tensor(compute_label_weights(train_labels), dtype=torch.float32, device=device)
    label_priors = torch.tensor(compute_label_priors(train_labels), dtype=torch.float32, device=device)
    adjacency_np = build_label_graph(train_labels)
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=device)
    np.save(output_dir / "label_adjacency.npy", adjacency_np)

    model = LQDDNModel(
        backbone_name=config["model"]["backbone"],
        num_labels=len(label_names),
        num_attention_heads=config["model"]["num_attention_heads"],
        dropout=config["model"]["dropout"],
        use_label_queries=config["model"]["use_label_queries"],
        use_static_graph=config["model"]["use_static_graph"],
        use_dynamic_dependency=config["model"]["use_dynamic_dependency"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    total_steps = len(dataloaders["train"]) * config["training"]["epochs"]
    warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.0))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = bool(config["training"].get("mixed_precision", False) and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    best_state_path = output_dir / "best_model.pt"
    best_score = float("-inf")
    patience_counter = 0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloaders["train"], desc=f"Epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch, adjacency)
                classification_loss = build_classification_loss(config, outputs.logits, batch["labels"], label_weights)
                ranking_loss = multilabel_ranking_loss(outputs.logits, batch["labels"], margin=config["loss"]["ranking_margin"])
                calibration_loss = rare_label_focal_loss(
                    outputs.logits,
                    batch["labels"],
                    label_weights=label_priors.reciprocal(),
                    gamma=config["loss"]["calibration_gamma"],
                )
                total_loss = (
                    classification_loss
                    + config["loss"]["lambda_ranking"] * ranking_loss
                    + config["loss"]["lambda_calibration"] * calibration_loss
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(total_loss.item())
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

        validation_arrays = collect_outputs(
            model,
            dataloaders["validation"],
            device,
            adjacency,
            desc=f"Validation {epoch}",
        )
        validation_probabilities = validation_arrays["probabilities"]
        validation_thresholds = None
        if config["inference"]["threshold_mode"] == "tuned":
            validation_thresholds = tune_thresholds(validation_arrays["targets"], validation_probabilities)
        validation_result = evaluate_arrays(
            validation_arrays["targets"],
            validation_arrays["logits"],
            validation_probabilities,
            label_names,
            threshold_mode=config["inference"]["threshold_mode"],
            fixed_threshold=config["inference"]["fixed_threshold"],
            thresholds=validation_thresholds,
        )
        validation_result["metrics"].update(
            compute_eval_losses(
                config,
                validation_arrays["logits"],
                validation_arrays["targets"],
                label_weights,
                label_priors,
            )
        )
        score = select_metric(validation_result["metrics"], config["training"]["monitor_metric"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(len(dataloaders["train"]), 1),
                "validation_score": score,
                "validation_macro_f1": validation_result["metrics"]["macro_f1"],
                "validation_map": validation_result["metrics"]["mAP"],
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={history[-1]['train_loss']:.4f} "
            f"validation_score={score:.4f} "
            f"{format_metrics(validation_result['metrics'])}"
        )
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), best_state_path)
            print(f"Saved new best model at epoch {epoch} with {config['training']['monitor_metric']}={score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(best_state_path, map_location=device))

    validation_arrays = collect_outputs(model, dataloaders["validation"], device, adjacency, desc="Validation Final")
    test_arrays = collect_outputs(model, dataloaders["test"], device, adjacency, desc="Test Final")
    calibrator = None
    if config["calibration"]["enabled"]:
        calibrator = fit_calibrator(
            validation_arrays["logits"],
            validation_arrays["targets"],
            max_iter=config["calibration"]["max_iter"],
            learning_rate=config["calibration"]["learning_rate"],
        )
        save_calibrator(calibrator, output_dir / "calibrator.json")

    validation_probabilities = apply_calibration(validation_arrays["probabilities"], validation_arrays["logits"], calibrator)
    test_probabilities = apply_calibration(test_arrays["probabilities"], test_arrays["logits"], calibrator)
    thresholds = None
    if config["inference"]["threshold_mode"] == "tuned":
        thresholds = tune_thresholds(validation_arrays["targets"], validation_probabilities)

    validation_result = evaluate_arrays(
        validation_arrays["targets"],
        validation_arrays["logits"],
        validation_probabilities,
        label_names,
        threshold_mode=config["inference"]["threshold_mode"],
        fixed_threshold=config["inference"]["fixed_threshold"],
        thresholds=thresholds,
    )
    validation_result["metrics"].update(
        compute_eval_losses(
            config,
            validation_arrays["logits"],
            validation_arrays["targets"],
            label_weights,
            label_priors,
        )
    )
    test_result = evaluate_arrays(
        test_arrays["targets"],
        test_arrays["logits"],
        test_probabilities,
        label_names,
        threshold_mode=config["inference"]["threshold_mode"],
        fixed_threshold=config["inference"]["fixed_threshold"],
        thresholds=thresholds,
    )
    test_result["metrics"].update(
        compute_eval_losses(
            config,
            test_arrays["logits"],
            test_arrays["targets"],
            label_weights,
            label_priors,
        )
    )

    save_json({"labels": label_names}, output_dir / "label_list.json")
    save_json({"history": history}, output_dir / "history.json")
    save_json(config, output_dir / "config.json")
    save_json(
        {
            "mode": config["inference"]["threshold_mode"],
            "fixed_threshold": config["inference"]["fixed_threshold"],
            "tuned_thresholds": thresholds,
        },
        output_dir / "thresholds.json",
    )
    model.encoder.config.save_pretrained(output_dir / "encoder_config")
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    payload = {
        "language": config["language"]["name"],
        "experiment_name": config["experiment_name"],
        "validation": validation_result["metrics"],
        "test": test_result["metrics"],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("Final validation metrics:", format_metrics(validation_result["metrics"]))
    print("Final test metrics:", format_metrics(test_result["metrics"]))
    return payload
