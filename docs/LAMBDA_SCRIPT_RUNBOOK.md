# Lambda Bash Script Runbook

This runbook is for `runs/lambda_full_pipeline.sh`.

## 1) Fresh start (recommended)

```bash
cd ~/agml_chat
git pull
rm -rf .venv
```

## 1b) Environment already set up (fast rerun)

Use this when `.venv` already exists and you just want to rerun training.

Strict FlashAttention (fails if not active):

```bash
cd ~/agml_chat
git pull
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
INSTALL_FLASH_ATTN=1 \
FLASH_ATTN_FORCE_BUILD=0 \
FLASH_ATTN_NO_DEPS=1 \
STRICT_FLASH_ATTN=1 \
FLASH_ATTN_MAX_JOBS=1 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
FLASH_ATTN_RETRY_MINIMAL=1 \
bash runs/lambda_full_pipeline.sh
```

Allow fallback to SDPA:

```bash
cd ~/agml_chat
git pull
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
INSTALL_FLASH_ATTN=1 \
FLASH_ATTN_FORCE_BUILD=0 \
FLASH_ATTN_NO_DEPS=1 \
STRICT_FLASH_ATTN=0 \
FLASH_ATTN_MAX_JOBS=1 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
bash runs/lambda_full_pipeline.sh
```

## 2) Strict FlashAttention run (fail if FlashAttention is not active)

```bash
cd ~/agml_chat
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
INSTALL_FLASH_ATTN=1 \
FLASH_ATTN_FORCE_BUILD=0 \
FLASH_ATTN_NO_DEPS=1 \
STRICT_FLASH_ATTN=1 \
FLASH_ATTN_MAX_JOBS=2 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
FLASH_ATTN_RETRY_MINIMAL=1 \
DATASETS=plant_village_classification \
TRAIN_RATIO=1.0 VAL_RATIO=0.0 TEST_RATIO=0.0 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
GRAD_ACCUM=1 \
LORA_R=64 \
LORA_ALPHA=128 \
LORA_DROPOUT=0.05 \
bash runs/lambda_full_pipeline.sh
```

If this fails, check:

```bash
tail -n 120 runs/*/logs/flash_attn_install.log
```

## 3) Fallback run (allow SDPA if FlashAttention cannot be built)

```bash
cd ~/agml_chat
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
INSTALL_FLASH_ATTN=1 \
FLASH_ATTN_FORCE_BUILD=0 \
FLASH_ATTN_NO_DEPS=1 \
STRICT_FLASH_ATTN=0 \
FLASH_ATTN_MAX_JOBS=2 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
bash runs/lambda_full_pipeline.sh
```

## 3b) Run explicitly without FlashAttention (force SDPA)

```bash
cd ~/agml_chat
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
NO_FLASH_ATTN=1 \
STRICT_FLASH_ATTN=0 \
bash runs/lambda_full_pipeline.sh
```

## 4) Verify what attention path was used

```bash
grep -n "attention implementation" runs/*/logs/train.log
```

Expected values:
- `flash_attention_2`: FlashAttention is active
- `sdpa`: fallback path
- `eager`: non-CUDA path

## 5) Optional toggles

- Disable LoRA entirely:

```bash
NO_LORA=1 bash runs/lambda_full_pipeline.sh
```

- Disable flash-attn intentionally:

```bash
NO_FLASH_ATTN=1 bash runs/lambda_full_pipeline.sh
```

- Update repo inside script before run:

```bash
UPDATE_REPO=1 GIT_REF=main bash runs/lambda_full_pipeline.sh
```

## 6) Script Flags Reference

Use these as environment variables before `bash runs/lambda_full_pipeline.sh`.

### Core

| Flag | Default | Description |
|---|---|---|
| `REPO_DIR` | current directory | Repo root to run from. |
| `UPDATE_REPO` | `0` | If `1`, runs `git fetch/pull` before pipeline. |
| `GIT_REF` | empty | Branch/tag/SHA to checkout when `UPDATE_REPO=1`. |
| `LOCK_MODE` | `frozen` | `frozen` uses lockfile, `refresh` relocks then syncs. |
| `MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF model id/path for SFT. |
| `DATASETS` | `plant_village_classification` | Comma-separated AgML datasets. |
| `PROMPT_CONFIG` | `configs/prompt_config.example.yaml` | Prompt config YAML path. |

### Dataset Prep

| Flag | Default | Description |
|---|---|---|
| `TRAIN_RATIO` | `1.0` | Train split ratio. |
| `VAL_RATIO` | `0.0` | Val split ratio. |
| `TEST_RATIO` | `0.0` | Test split ratio. |
| `SEED` | `42` | Split and sampling seed. |
| `MAX_SAMPLES_PER_DATASET` | empty | Optional cap per dataset before splitting. |

### Training

| Flag | Default | Description |
|---|---|---|
| `EPOCHS` | `1` | Number of training epochs. |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | Per-device train batch size. |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps. |
| `LOGGING_STEPS` | `5` | Trainer logging interval. |
| `SAVE_STEPS` | `100` | Checkpoint save interval. |
| `MAX_LENGTH` | `2048` | Token sequence length for processor/training. |
| `LEARNING_RATE` | `2e-5` | Learning rate. |
| `WARMUP_RATIO` | `0.03` | LR warmup ratio. |

### LoRA

| Flag | Default | Description |
|---|---|---|
| `NO_LORA` | `0` | If `1`, disables LoRA and runs full-parameter training path. |
| `LORA_R` | `16` | LoRA rank. |
| `LORA_ALPHA` | `32` | LoRA alpha. |
| `LORA_DROPOUT` | `0.05` | LoRA dropout. |
| `LORA_TARGET_MODULES` | empty | Comma-separated target modules (optional override). |

### Torch/Torchvision Repair

| Flag | Default | Description |
|---|---|---|
| `AUTO_FIX_TORCH_STACK` | `1` | Auto-repair torch/torchvision mismatch if detected. |
| `GPU_WHEEL_TAG` | `auto` | CUDA wheel tag (`auto`, `cu128`, `cu130`, ...). |
| `GPU_TORCH_VERSION` | `2.11.0` | Torch version used by repair path. |
| `GPU_TORCHVISION_VERSION` | `0.26.0` | Torchvision version used by repair path. |

### FlashAttention

| Flag | Default | Description |
|---|---|---|
| `NO_FLASH_ATTN` | `0` | If `1`, disables flash path and forces SDPA. |
| `INSTALL_FLASH_ATTN` | `1` | Attempt flash-attn install before training. |
| `STRICT_FLASH_ATTN` | `1` | If `1`, fail if flash-attn is unavailable or not used. |
| `FLASH_ATTN_FORCE_BUILD` | `0` | If `1`, skip wheel-first and force source build. |
| `FLASH_ATTN_VERSION` | empty | Optional flash-attn version pin. |
| `FLASH_ATTN_NO_DEPS` | `1` | Build flash-attn without reinstalling torch deps. |
| `FLASH_ATTN_MAX_JOBS` | `24` | Build parallelism (`MAX_JOBS`). |
| `FLASH_ATTN_NVCC_THREADS` | `1` | NVCC thread count used by build. |
| `FLASH_ATTN_TORCH_CUDA_ARCH_LIST` | `9.0` | CUDA arch list for compile. |
| `FLASH_ATTN_RETRY_MINIMAL` | `1` | Retry source build with minimal parallelism if first build fails. |

### Output and Serving

| Flag | Default | Description |
|---|---|---|
| `RUN_TAG` | timestamp | Tag used for output directory names. |
| `DATA_DIR` | `data/agml_sft_<RUN_TAG>` | Prepared dataset output dir. |
| `RUN_DIR` | `runs/sft_<RUN_TAG>` | Training output dir. |
| `LOG_DIR` | `<RUN_DIR>/logs` | Logs directory. |
| `START_WEB` | `0` | If `1`, launch web chat server after training. |
| `HOST` | `0.0.0.0` | Web server host. |
| `PORT` | `8000` | Web server port. |

### Example: Bigger Batch + LoRA

```bash
cd ~/agml_chat
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
GRAD_ACCUM=1 \
LORA_R=64 \
LORA_ALPHA=128 \
LORA_DROPOUT=0.05 \
bash runs/lambda_full_pipeline.sh
```

### Batch Size Defaults and How To Change

Defaults in `runs/lambda_full_pipeline.sh`:
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRAD_ACCUM=8`

Effective batch size is:
- `effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACCUM`
- default effective batch = `8`

Examples:

```bash
# Increase per-device batch, no accumulation
PER_DEVICE_TRAIN_BATCH_SIZE=8 GRAD_ACCUM=1 bash runs/lambda_full_pipeline.sh

# Keep flash-attn off and use bigger batch
NO_FLASH_ATTN=1 PER_DEVICE_TRAIN_BATCH_SIZE=8 GRAD_ACCUM=1 bash runs/lambda_full_pipeline.sh
```

### Example: Force source build only when wheel-first fails

```bash
cd ~/agml_chat
FLASH_ATTN_FORCE_BUILD=1 \
FLASH_ATTN_MAX_JOBS=1 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
bash runs/lambda_full_pipeline.sh
```

## 7) Common troubleshooting

- Torch/Torchvision mismatch: keep `AUTO_FIX_TORCH_STACK=1 GPU_WHEEL_TAG=auto`.
- Missing cuDNN/other CUDA runtime libs: rerun with fresh `.venv` and strict command above.
- FlashAttention build errors: inspect `runs/*/logs/flash_attn_install.log` and lower `FLASH_ATTN_MAX_JOBS` if OOM during build.
- If `flash_attn_install.log` is very long, check the final result quickly with:

```bash
grep -n "attention implementation" runs/*/logs/train.log
```

## 8) Upload Trained Model To Hugging Face

The pipeline outputs the final model at:
- `runs/sft_<RUN_TAG>/final`

If `START_WEB=1`, the script already launches the chat UI using that exact folder.

### 8a) Set token

```bash
export HF_TOKEN=hf_xxx_your_token
```

### 8b) Upload full final model folder

```bash
cd ~/agml_chat
.venv/bin/python -m scripts.upload_to_hf \
  --model-dir runs/sft_<RUN_TAG>/final \
  --repo-id <your-username-or-org>/<repo-name> \
  --private \
  --commit-message "Upload agml-chat fine-tuned model"
```

### 8c) Optional: upload only safetensors + config files

```bash
cd ~/agml_chat
.venv/bin/python -m scripts.upload_to_hf \
  --model-dir runs/sft_<RUN_TAG>/final \
  --repo-id <your-username-or-org>/<repo-name> \
  --allow-patterns "config.json,tokenizer.json,tokenizer_config.json,preprocessor_config.json,*.safetensors,*.model,*.txt" \
  --commit-message "Upload compact model artifacts"
```
