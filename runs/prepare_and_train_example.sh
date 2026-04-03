#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end pipeline.
# Edit model and dataset list to your preference.
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASETS="plant_village_classification,rice_leaf_disease_classification"

uv run -m scripts.prepare_agml_sft \
  --datasets "$DATASETS" \
  --output-dir data/agml_sft \
  --prompt-config configs/prompt_config.example.yaml

uv run -m scripts.chat_sft \
  --model-name "$MODEL" \
  --train-jsonl data/agml_sft/train.jsonl \
  --val-jsonl data/agml_sft/val.jsonl \
  --output-dir runs/sft_qwen25vl3b \
  --epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
