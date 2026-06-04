# LQDDN: Label-Query Dynamic Dependency Network for Multilabel Emotion Recognition

This repository contains the official implementation of **LQDDN (Label-Query Dynamic Dependency Network)** for multilabel emotion recognition.

LQDDN is designed to address three common difficulties in multilabel emotion classification:

- modeling dependencies among correlated emotion labels
- improving label ranking quality
- producing more reliable predictions for rare labels

The framework combines a pretrained transformer encoder, a label-query decoder, a static label co-occurrence graph, a dynamic label interaction branch, and ranking- and calibration-aware learning.

## Method Summary

The proposed framework consists of the following components:

1. **Transformer encoder**
   contextualizes the input text using a pretrained language model.
2. **Label-query decoder**
   assigns one learnable query to each emotion label and extracts label-specific evidence through cross-attention.
3. **Dual dependency modeling**
   combines a static co-occurrence graph prior with dynamic self-attention among label representations.
4. **Adaptive fusion**
   balances global label priors and instance-specific label interactions.
5. **Label-specific prediction**
   predicts multilabel emotion probabilities using dedicated decision functions for each label.
6. **Threshold tuning and calibration**
   uses validation-tuned thresholds and optional post-hoc calibration for more reliable final decisions.

## Repository Structure

```text
.
├── configs/
│   ├── english_lqddn.yaml
│   └── chinese_lqddn.yaml
├── lqddn/
│   ├── analysis.py
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── hf_utils.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── trainer.py
├── scripts/
│   ├── aggregate_results.py
│   ├── analyze_lqddn.py
│   ├── evaluate_lqddn.py
│   ├── run_ablations.py
│   └── train_lqddn.py
├── requirements.txt
└── run.sh
```

## Datasets

This codebase assumes the following dataset layout:

- `English/train.txt`
- `English/validation.txt`
- `English/test.txt`
- `Chinese/train.csv`
- `Chinese/validation.csv`
- `Chinese/test.csv`

The loader automatically infers:

- the text column
- the multilabel emotion columns
- binary-column or label-string format

## Environment

The implementation was developed with:

- Python `3.13`
- PyTorch `2.8.0`
- Hugging Face `transformers`

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Default Backbones

- **English:** `roberta-base`
- **Chinese:** `hfl/chinese-roberta-wwm-ext`

These can be changed from the config files or overridden from the command line.

## Training

Train the full model on English:

```bash
python scripts/train_lqddn.py --config configs/english_lqddn.yaml --ablation A5_full_lqddn
```

Train the full model on Chinese:

```bash
python scripts/train_lqddn.py --config configs/chinese_lqddn.yaml --ablation A5_full_lqddn
```

## Ablations

The current ablation suite includes:

- `A5_full_lqddn`
- `A6_full_no_graph`
- `A7_full_no_dynamic`
- `A8_full_no_ranking`
- `A9_full_no_calibration`

Run the full ablation suite for English:

```bash
python scripts/run_ablations.py --config configs/english_lqddn.yaml
```

Run the full ablation suite for Chinese:

```bash
python scripts/run_ablations.py --config configs/chinese_lqddn.yaml
```

## One-Command Runner

Run the full model first and then the remaining ablations:

```bash
./run.sh english
./run.sh chinese
./run.sh both
```

## Evaluate Saved Checkpoints

You can evaluate an already trained checkpoint without retraining:

```bash
python scripts/evaluate_lqddn.py \
  --config configs/english_lqddn.yaml \
  --model outputs/english/A5_full_lqddn/best_model.pt
```

```bash
python scripts/evaluate_lqddn.py \
  --config configs/chinese_lqddn.yaml \
  --model outputs/chinese/A5_full_lqddn/best_model.pt
```

## Interpretability and Analysis

The repository includes an analysis pipeline for:

- case studies
- error analysis
- cross-attention heatmaps
- dynamic label interaction heatmaps
- fusion gate summaries
- static graph visualization

Example:

```bash
python scripts/analyze_lqddn.py \
  --config configs/english_lqddn.yaml \
  --model outputs/english/A5_full_lqddn/best_model.pt \
  --split test \
  --output-dir outputs/english/A5_full_lqddn/analysis
```

The script generates files such as:

- `case_studies.json`
- `error_analysis.csv`
- `analysis_summary.json`
- `heatmaps/static_label_graph.png`
- `heatmaps/*_cross_attention.png`
- `heatmaps/*_dynamic_attention.png`
- `heatmaps/*_fusion_gates_average.png`

## Result Aggregation

Aggregate experiment outputs with:

```bash
python scripts/aggregate_results.py --root outputs
```

This produces:

- `results_summary.csv`
- `per_label_results.csv`
- `ablation_table.csv`

## Output Layout

```text
outputs/
  english/
    A5_full_lqddn/
      best_model.pt
      calibrator.json
      config.json
      encoder_config/
      history.json
      label_adjacency.npy
      label_list.json
      metrics.json
      thresholds.json
      tokenizer/
  chinese/
    A5_full_lqddn/
      ...
```

## Notes

- English and Chinese are trained and evaluated separately.
- The code saves ranking loss, classification loss, calibration loss, and total loss in evaluation metrics.
- Hugging Face safetensors auto-conversion is disabled in-process to avoid noisy background warnings for repositories with disabled discussions.