#!/usr/bin/env bash
set -euo pipefail

# End-to-end remote pipeline for Lambda Labs GPU boxes.
#
# Usage:
#   bash runs/lambda_full_pipeline.sh
#
# Optional overrides:
#   MODEL="Qwen/Qwen2.5-VL-3B-Instruct" DATASETS="plant_village_classification" bash runs/lambda_full_pipeline.sh
#   TRAIN_RATIO=1.0 VAL_RATIO=0.0 TEST_RATIO=0.0 bash runs/lambda_full_pipeline.sh
#   START_WEB=1 HOST=0.0.0.0 PORT=8000 bash runs/lambda_full_pipeline.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
UPDATE_REPO="${UPDATE_REPO:-0}"
GIT_REF="${GIT_REF:-}"
LOCK_MODE="${LOCK_MODE:-frozen}" # frozen|refresh
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASETS="${DATASETS:-plant_village_classification}"
PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompt_config.example.yaml}"

# Split defaults are train-only to keep the path robust if you evaluate elsewhere.
TRAIN_RATIO="${TRAIN_RATIO:-1.0}"
VAL_RATIO="${VAL_RATIO:-0.0}"
TEST_RATIO="${TEST_RATIO:-0.0}"
SEED="${SEED:-42}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"

EPOCHS="${EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

NO_LORA="${NO_LORA:-0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
NO_FLASH_ATTN="${NO_FLASH_ATTN:-0}"

START_WEB="${START_WEB:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/agml_sft_${RUN_TAG}}"
RUN_DIR="${RUN_DIR:-runs/sft_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

if [[ "${UPDATE_REPO}" == "1" ]]; then
  echo "[0/7] Updating repository"
  git fetch --all --prune
  if [[ -n "${GIT_REF}" ]]; then
    git checkout "${GIT_REF}"
  fi
  git pull --ff-only
fi

echo "[1/7] Checking GPU availability"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (continuing)."
fi

echo "[2/7] Ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[3/7] Syncing environment"
if [[ "${LOCK_MODE}" == "refresh" ]]; then
  uv lock
  uv sync --extra gpu --group dev
else
  uv sync --frozen --extra gpu --group dev
fi

echo "[4/7] Verifying runtime deps (torch/torchvision)"
uv run python - <<'PY'
import torch
import torchvision
print(f"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.cuda.is_available()}")
PY

if [[ ! -f "${PROMPT_CONFIG}" ]]; then
  echo "Prompt config not found: ${PROMPT_CONFIG}" >&2
  exit 1
fi

echo "[5/7] Preparing AgML SFT dataset"
PREP_CMD=(
  uv run -m scripts.prepare_agml_sft
  --datasets "${DATASETS}"
  --output-dir "${DATA_DIR}"
  --prompt-config "${PROMPT_CONFIG}"
  --train-ratio "${TRAIN_RATIO}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --seed "${SEED}"
)

if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
  PREP_CMD+=(--max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}")
fi

"${PREP_CMD[@]}" | tee "${LOG_DIR}/prepare.log"

TRAIN_JSONL="${DATA_DIR}/train.jsonl"
VAL_JSONL="${DATA_DIR}/val.jsonl"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "train.jsonl was not generated at ${TRAIN_JSONL}" >&2
  exit 1
fi

echo "[6/7] Starting fine-tuning"
mkdir -p "${RUN_DIR}"

TRAIN_CMD=(
  uv run -m scripts.chat_sft
  --model-name "${MODEL}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${RUN_DIR}"
  --device cuda
  --epochs "${EPOCHS}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --logging-steps "${LOGGING_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
)

if [[ -f "${VAL_JSONL}" ]]; then
  TRAIN_CMD+=(--val-jsonl "${VAL_JSONL}")
fi

if [[ "${NO_LORA}" == "1" ]]; then
  TRAIN_CMD+=(--no-lora)
fi

if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  TRAIN_CMD+=(--lora-target-modules "${LORA_TARGET_MODULES}")
fi

if [[ "${NO_FLASH_ATTN}" == "1" ]]; then
  TRAIN_CMD+=(--no-flash-attn)
fi

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

FINAL_MODEL_DIR="${RUN_DIR}/final"
if [[ ! -d "${FINAL_MODEL_DIR}" ]]; then
  echo "Training finished but final model directory not found: ${FINAL_MODEL_DIR}" >&2
  exit 1
fi

echo "[7/7] Pipeline complete"
echo "Dataset dir: ${DATA_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Train log: ${LOG_DIR}/train.log"
echo "Final model: ${FINAL_MODEL_DIR}"

if [[ "${START_WEB}" == "1" ]]; then
  echo "Starting web server on ${HOST}:${PORT} ..."
  exec uv run -m scripts.chat_web \
    --model "${FINAL_MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
fi
