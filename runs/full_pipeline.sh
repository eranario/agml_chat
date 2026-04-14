#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash runs/full_pipeline.sh
#
# Optional overrides:
#   MODEL="Qwen/Qwen2.5-VL-3B-Instruct" DATASETS="plant_village_classification" bash runs/full_pipeline.sh
#   TRAIN_RATIO=1.0 VAL_RATIO=0.0 TEST_RATIO=0.0 bash runs/full_pipeline.sh
#   LIVE_METRICS=1 LIVE_METRICS_EVERY_N_LOGS=1 bash runs/full_pipeline.sh
#   START_WEB=1 HOST=0.0.0.0 PORT=8000 bash runs/full_pipeline.sh
#   AUTO_FIX_TORCH_STACK=1 GPU_WHEEL_TAG=auto GPU_TORCH_VERSION=2.11.0 GPU_TORCHVISION_VERSION=0.26.0 bash runs/full_pipeline.sh
#   INSTALL_FLASH_ATTN=1 STRICT_FLASH_ATTN=1 FLASH_ATTN_FORCE_BUILD=0 FLASH_ATTN_NO_DEPS=1 bash runs/full_pipeline.sh
#   GEMMA4_TRANSFORMERS_SOURCE=1 MODEL=google/gemma-4-E2B-it bash runs/full_pipeline.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
UPDATE_REPO="${UPDATE_REPO:-0}"
GIT_REF="${GIT_REF:-}"
LOCK_MODE="${LOCK_MODE:-frozen}" # frozen|refresh
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}" # If 1, skips uv sync, pip installs, and torch repair
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASETS="${DATASETS:-plant_village_classification}"
PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompt_config.example.yaml}"
AUTO_FIX_TORCH_STACK="${AUTO_FIX_TORCH_STACK:-1}"
GPU_WHEEL_TAG="${GPU_WHEEL_TAG:-auto}" # auto|cu128|cu130|...
GPU_TORCH_VERSION="${GPU_TORCH_VERSION:-2.11.0}"
GPU_TORCHVISION_VERSION="${GPU_TORCHVISION_VERSION:-0.26.0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
STRICT_FLASH_ATTN="${STRICT_FLASH_ATTN:-1}"
FLASH_ATTN_FORCE_BUILD="${FLASH_ATTN_FORCE_BUILD:-0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-}" # optional pin, e.g. 2.8.3
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-24}"
FLASH_ATTN_NO_DEPS="${FLASH_ATTN_NO_DEPS:-1}"
FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-1}"
FLASH_ATTN_TORCH_CUDA_ARCH_LIST="${FLASH_ATTN_TORCH_CUDA_ARCH_LIST:-9.0}"
FLASH_ATTN_RETRY_MINIMAL="${FLASH_ATTN_RETRY_MINIMAL:-1}"
GEMMA4_TRANSFORMERS_SOURCE="${GEMMA4_TRANSFORMERS_SOURCE:-auto}" # auto|1|0

# Split defaults create both train and validation data for in-run eval/monitoring.
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.0}"
SEED="${SEED:-42}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

EPOCHS="${EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

NO_LORA="${NO_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
NO_FLASH_ATTN="${NO_FLASH_ATTN:-0}"
RUN_EVAL_INFERENCE="${RUN_EVAL_INFERENCE:-0}"
NO_METRICS_EXPORT="${NO_METRICS_EXPORT:-0}"
LIVE_METRICS="${LIVE_METRICS:-1}"
LIVE_METRICS_EVERY_N_LOGS="${LIVE_METRICS_EVERY_N_LOGS:-1}"

START_WEB="${START_WEB:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/agml_sft_${RUN_TAG}}"
RUN_DIR="${RUN_DIR:-runs/sft_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
FLASH_ATTN_LOG="${LOG_DIR}/flash_attn_install.log"

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

if [[ "${SKIP_ENV_SETUP}" != "1" ]]; then
  echo "[3/7] Syncing environment"
  if [[ "${LOCK_MODE}" == "refresh" ]]; then
    uv lock
    uv sync --extra gpu --group dev
  else
    uv sync --frozen --extra gpu --group dev
  fi

  if [[ "${GEMMA4_TRANSFORMERS_SOURCE}" == "1" || ( "${GEMMA4_TRANSFORMERS_SOURCE}" == "auto" && "${MODEL,,}" == *"gemma-4"* ) ]]; then
    echo "[3b/7] Installing latest Transformers from source for Gemma 4 compatibility"
    uv pip install --python "${REPO_DIR}/.venv/bin/python" --upgrade "git+https://github.com/huggingface/transformers.git"
  fi
fi

VENV_PY="${REPO_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Expected virtualenv python not found at ${VENV_PY}" >&2
  exit 1
fi

echo "[4/7] Verifying runtime deps (torch/torchvision)"
verify_torch_stack() {
  "${VENV_PY}" - <<'PY'
import torch
import torchvision
print(
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"torchvision={torchvision.__version__} cuda_available={torch.cuda.is_available()}"
)
PY
}

is_flash_attn_installed() {
  "${VENV_PY}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("flash_attn") is not None else "0")
PY
}

try_install_flash_attn() {
  local pkg
  pkg="flash-attn"
  if [[ -n "${FLASH_ATTN_VERSION}" ]]; then
    pkg="flash-attn==${FLASH_ATTN_VERSION}"
  fi

  : > "${FLASH_ATTN_LOG}"
  
  if [[ -z "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
    echo "Resolving pre-built flash-attn wheel URL..." | tee -a "${FLASH_ATTN_LOG}"
    FLASH_ATTN_WHEEL_URL="$("${VENV_PY}" scripts/resolve_flash_attn_wheel.py 2>/dev/null || true)"
  fi
  
  if [[ -n "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
    echo "Installing flash-attn from pre-built wheel URL: ${FLASH_ATTN_WHEEL_URL}" | tee -a "${FLASH_ATTN_LOG}"
    if uv pip install --python "${VENV_PY}" "${FLASH_ATTN_WHEEL_URL}" >>"${FLASH_ATTN_LOG}" 2>&1; then
      return 0
    fi
    echo "Pre-built wheel installation failed. Falling back..." | tee -a "${FLASH_ATTN_LOG}"
  fi

  if [[ "${FLASH_ATTN_FORCE_BUILD}" == "1" ]]; then
    echo "Installing ${pkg} (source-build mode) ..." | tee -a "${FLASH_ATTN_LOG}"
  else
    echo "Installing ${pkg} (wheel-first strategy) ..." | tee -a "${FLASH_ATTN_LOG}"
    if MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" uv pip install --python "${VENV_PY}" --no-build-isolation "${pkg}" >>"${FLASH_ATTN_LOG}" 2>&1; then
      return 0
    fi
    echo "Wheel install failed. See ${FLASH_ATTN_LOG}" | tee -a "${FLASH_ATTN_LOG}"
  fi

  echo "Trying source build with --no-build-isolation ..." | tee -a "${FLASH_ATTN_LOG}"
  echo "Build env: MAX_JOBS=${FLASH_ATTN_MAX_JOBS} NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS} TORCH_CUDA_ARCH_LIST=${FLASH_ATTN_TORCH_CUDA_ARCH_LIST}" | tee -a "${FLASH_ATTN_LOG}"
  uv pip install --python "${VENV_PY}" -U ninja packaging setuptools wheel psutil >>"${FLASH_ATTN_LOG}" 2>&1 || true
  local -a source_cmd
  source_cmd=(
    uv pip install
    --python "${VENV_PY}"
    --no-build-isolation
    --force-reinstall
  )
  if [[ "${FLASH_ATTN_NO_DEPS}" == "1" ]]; then
    source_cmd+=(--no-deps)
  fi
  source_cmd+=("${pkg}")

  if MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
    NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
    TORCH_CUDA_ARCH_LIST="${FLASH_ATTN_TORCH_CUDA_ARCH_LIST}" \
    "${source_cmd[@]}" >>"${FLASH_ATTN_LOG}" 2>&1; then
    return 0
  fi

  if [[ "${FLASH_ATTN_RETRY_MINIMAL}" == "1" ]]; then
    echo "Retrying source build with minimal parallelism ..." | tee -a "${FLASH_ATTN_LOG}"
    if MAX_JOBS="1" \
      NVCC_THREADS="1" \
      TORCH_CUDA_ARCH_LIST="${FLASH_ATTN_TORCH_CUDA_ARCH_LIST}" \
      "${source_cmd[@]}" >>"${FLASH_ATTN_LOG}" 2>&1; then
      return 0
    fi
  fi

  echo "Source build failed. See ${FLASH_ATTN_LOG}" | tee -a "${FLASH_ATTN_LOG}"
  return 1
}

flash_attn_version() {
  "${VENV_PY}" - <<'PY'
try:
    import flash_attn
    print(getattr(flash_attn, "__version__", "unknown"))
except Exception:
    print("")
PY
}

detect_cuda_wheel_tag_from_torch() {
  "${VENV_PY}" - <<'PY'
try:
    import torch
except Exception:
    print("")
    raise SystemExit(0)

cuda = torch.version.cuda
if not cuda:
    print("")
else:
    parts = cuda.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        print(f"cu{parts[0]}{parts[1]}")
    else:
        print("")
PY
}

if ! verify_torch_stack; then
  echo "Torch/Torchvision runtime check failed."
  if [[ "${AUTO_FIX_TORCH_STACK}" == "1" ]]; then
    REPAIR_WHEEL_TAG="${GPU_WHEEL_TAG}"
    if [[ "${REPAIR_WHEEL_TAG}" == "auto" ]]; then
      DETECTED_TAG="$(detect_cuda_wheel_tag_from_torch || true)"
      if [[ -n "${DETECTED_TAG}" ]]; then
        REPAIR_WHEEL_TAG="${DETECTED_TAG}"
      else
        REPAIR_WHEEL_TAG="cu128"
      fi
    fi

    echo "Attempting repair with matching CUDA wheels (${REPAIR_WHEEL_TAG}) ..."
    uv pip uninstall --python "${VENV_PY}" torch torchvision || true
    uv pip install --python "${VENV_PY}" \
      --index-url "https://download.pytorch.org/whl/${REPAIR_WHEEL_TAG}" \
      "torch==${GPU_TORCH_VERSION}+${REPAIR_WHEEL_TAG}" \
      "torchvision==${GPU_TORCHVISION_VERSION}+${REPAIR_WHEEL_TAG}"
    verify_torch_stack
  else
    echo "Set AUTO_FIX_TORCH_STACK=1 to auto-repair torch/torchvision mismatch." >&2
    exit 1
  fi
fi

if [[ "${NO_FLASH_ATTN}" != "1" ]]; then
  echo "[4b/7] Checking flash-attn availability"
  FLASH_PRESENT="$(is_flash_attn_installed)"
  if [[ "${FLASH_PRESENT}" == "1" ]]; then
    echo "flash-attn already installed (version: $(flash_attn_version))."
  elif [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    echo "flash-attn not found; attempting installation ..."
    if try_install_flash_attn; then
      FLASH_PRESENT="$(is_flash_attn_installed)"
      if [[ "${FLASH_PRESENT}" == "1" ]]; then
        echo "flash-attn installation succeeded (version: $(flash_attn_version))."
      fi
    fi
  fi

  if [[ "${FLASH_PRESENT}" != "1" ]]; then
    if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
      echo "flash-attn is required (STRICT_FLASH_ATTN=1) but is not installed." >&2
      if [[ -f "${FLASH_ATTN_LOG}" ]]; then
        echo "--- tail ${FLASH_ATTN_LOG} ---" >&2
        tail -n 80 "${FLASH_ATTN_LOG}" >&2 || true
      fi
      exit 1
    fi
    echo "flash-attn unavailable; training will use SDPA fallback."
  fi
fi

if [[ ! -f "${PROMPT_CONFIG}" ]]; then
  echo "Prompt config not found: ${PROMPT_CONFIG}" >&2
  exit 1
fi

echo "[5/7] Preparing AgML SFT dataset"
PREP_CMD=(
  "${VENV_PY}" -m scripts.prepare_agml_sft
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

if [[ "${SPECIES_SPECIFIC_OPTIONS:-1}" == "0" ]]; then
  PREP_CMD+=(--no-species-specific-options)
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
  "${VENV_PY}" -m scripts.chat_sft
  --model-name "${MODEL}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${RUN_DIR}"
  --device cuda
  --epochs "${EPOCHS}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --logging-steps "${LOGGING_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi

if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi

if [[ -f "${VAL_JSONL}" ]]; then
  TRAIN_CMD+=(--val-jsonl "${VAL_JSONL}")
fi

if [[ "${NO_LORA}" == "1" ]]; then
  TRAIN_CMD+=(--no-lora)
fi

if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  TRAIN_CMD+=(--lora-target-modules "${LORA_TARGET_MODULES}")
fi

LATEST_CHECKPOINT=$(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
  echo "Found existing checkpoint: ${LATEST_CHECKPOINT} - Resuming training automatically."
  
  if [[ "${SOFT_RESUME:-0}" == "1" ]]; then
    if [[ -f "${LATEST_CHECKPOINT}/optimizer.pt" ]]; then
      echo "SOFT_RESUME=1: Moving optimizer.pt to optimizer.pt.bak to avoid state dict mismatch."
      mv "${LATEST_CHECKPOINT}/optimizer.pt" "${LATEST_CHECKPOINT}/optimizer.pt.bak"
    fi
    if [[ -f "${LATEST_CHECKPOINT}/scaler.pt" ]]; then
      echo "SOFT_RESUME=1: Moving scaler.pt to scaler.pt.bak."
      mv "${LATEST_CHECKPOINT}/scaler.pt" "${LATEST_CHECKPOINT}/scaler.pt.bak"
    fi
    if [[ -f "${LATEST_CHECKPOINT}/scheduler.pt" ]]; then
      echo "SOFT_RESUME=1: Moving scheduler.pt to scheduler.pt.bak."
      mv "${LATEST_CHECKPOINT}/scheduler.pt" "${LATEST_CHECKPOINT}/scheduler.pt.bak"
    fi
  fi
  
  TRAIN_CMD+=(--resume-from-checkpoint "${LATEST_CHECKPOINT}")
fi

if [[ "${NO_FLASH_ATTN}" == "1" ]]; then
  TRAIN_CMD+=(--no-flash-attn)
fi

if [[ "${RUN_EVAL_INFERENCE}" == "1" ]]; then
  TRAIN_CMD+=(--run-eval-inference)
fi

if [[ "${NO_METRICS_EXPORT}" == "1" ]]; then
  TRAIN_CMD+=(--no-metrics-export)
fi

if [[ "${LIVE_METRICS}" == "1" && "${NO_METRICS_EXPORT}" != "1" ]]; then
  TRAIN_CMD+=(--live-metrics --live-metrics-every-n-logs "${LIVE_METRICS_EVERY_N_LOGS}")
fi

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

if [[ "${NO_FLASH_ATTN}" != "1" && "${STRICT_FLASH_ATTN}" == "1" ]]; then
  if ! grep -q "attention implementation 'flash_attention_2'" "${LOG_DIR}/train.log"; then
    echo "Expected flash_attention_2 but training did not use it. See ${LOG_DIR}/train.log" >&2
    exit 1
  fi
fi

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
  exec "${VENV_PY}" -m scripts.chat_web \
    --model "${FINAL_MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
fi
