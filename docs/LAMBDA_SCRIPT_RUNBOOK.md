# Lambda Bash Script Runbook

This runbook is for `runs/lambda_full_pipeline.sh`.

## 1) Fresh start (recommended)

```bash
cd ~/agml_chat
git pull
rm -rf .venv
```

## 2) Strict FlashAttention run (fail if FlashAttention is not active)

```bash
cd ~/agml_chat
LOCK_MODE=frozen \
AUTO_FIX_TORCH_STACK=1 \
GPU_WHEEL_TAG=auto \
INSTALL_FLASH_ATTN=1 \
FLASH_ATTN_FORCE_BUILD=1 \
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
FLASH_ATTN_FORCE_BUILD=1 \
FLASH_ATTN_NO_DEPS=1 \
STRICT_FLASH_ATTN=0 \
FLASH_ATTN_MAX_JOBS=2 \
FLASH_ATTN_NVCC_THREADS=1 \
FLASH_ATTN_TORCH_CUDA_ARCH_LIST=9.0 \
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

## 6) Common troubleshooting

- Torch/Torchvision mismatch: keep `AUTO_FIX_TORCH_STACK=1 GPU_WHEEL_TAG=auto`.
- Missing cuDNN/other CUDA runtime libs: rerun with fresh `.venv` and strict command above.
- FlashAttention build errors: inspect `runs/*/logs/flash_attn_install.log` and lower `FLASH_ATTN_MAX_JOBS` if OOM during build.
- If `flash_attn_install.log` is very long, check the final result quickly with:

```bash
grep -n "attention implementation" runs/*/logs/train.log
```
