#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

run_language() {
  local language="$1"
  local config_path

  case "$language" in
    english)
      config_path="configs/english_lqddn.yaml"
      ;;
    chinese)
      config_path="configs/chinese_lqddn.yaml"
      ;;
    *)
      echo "Unsupported language: $language" >&2
      exit 1
      ;;
  esac

  echo "Running full LQDDN model for ${language}..."
  "$PYTHON_BIN" scripts/train_lqddn.py \
    --config "$config_path" \
    --ablation A5_full_lqddn \
    --output-dir "${OUTPUT_ROOT}/${language}/A5_full_lqddn"

  echo "Running ablations for ${language}..."
  "$PYTHON_BIN" scripts/run_ablations.py \
    --config "$config_path" \
    --python "$PYTHON_BIN" \
    --output-root "$OUTPUT_ROOT"
}

main() {
  local target="${1:-both}"
  case "$target" in
    english)
      run_language "english"
      ;;
    chinese)
      run_language "chinese"
      ;;
    both)
      run_language "english"
      run_language "chinese"
      ;;
    *)
      echo "Usage: ./run.sh [english|chinese|both]" >&2
      exit 1
      ;;
  esac
}

main "$@"
