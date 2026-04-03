#!/usr/bin/env bash
set -euo pipefail

# Minimal CPU/MPS-friendly smoke path for agml-chat.
# Assumes you created an environment with `uv sync --extra cpu`.

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
IMAGE="${IMAGE:-}"
PROMPT="${PROMPT:-Classify the crop condition in this image.}"

if [[ -z "${IMAGE}" ]]; then
  echo "Set IMAGE=/path/to/image before running this script."
  exit 1
fi

uv run -m scripts.chat_cli \
  --model "${MODEL}" \
  --device cpu \
  --temperature 0.0 \
  --max-new-tokens 128 \
  --image "${IMAGE}" \
  --single-prompt "${PROMPT}"
