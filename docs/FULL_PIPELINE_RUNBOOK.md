# Gemma 4 Fine-Tuning Runbook

This runbook describes how to execute the SFT pipeline specifically tailored for tuning `google/gemma-4-E2B-it` via our orchestrator script `runs/full_pipeline.sh`.

## 1. Fresh Start (Standard Run)

To start a fresh training run for Gemma 4 using the default dataset (`plant_village_classification`):

```bash
MODEL="google/gemma-4-E2B-it" bash runs/full_pipeline.sh
```

**What this does:**
1. Automatically spins up/syncs `uv` virtual environments based on `pyproject.toml`.
2. Resolves and reinstalls accurate `torch`/`torchvision` bindings for your host GPU out-of-band.
3. Dynamically identifies your CUDA/PyTorch combination and installs the fastest *pre-built* **Flash Attention 2** wheel (no slow builds!).
4. Prepares the AgML dataset (e.g. enforcing species-specific options dynamically).
5. Tunes the model using standard parameters (LoRA R=16, Alpha=32).

---

## 2. Fast Development Rerun

If your environment (`.venv`) is already fully compiled and ready, and you just want to rerun training extremely quickly:

```bash
MODEL="google/gemma-4-E2B-it" SKIP_ENV_SETUP=1 bash runs/full_pipeline.sh
```

---

## 3. Dealing with Crashes (OOM / Soft Resume)

If a training job was preempted, crashed, or encountered PyTorch's infamous `optimizer.pt` bloat/mis-dimension error on resume, use the `SOFT_RESUME=1` environment variable.

Just match your exact param flags, provide the original `RUN_TAG`, and skip the env synchronization:

```bash
RUN_TAG="20260414_030352" \
MODEL="google/gemma-4-E2B-it" \
SKIP_ENV_SETUP=1 \
SOFT_RESUME=1 \
bash runs/full_pipeline.sh
```
> *Note: `SOFT_RESUME=1` safely isolates `optimizer.pt` before the Hugging Face Trainer boots it up, permitting dimension recovery and avoiding VRAM death spikes.*

---

## 4. Hyperparameter Sweeps

To execute a matrix grid search automatically iterating through varying combinations of Learning Rates and LoRA Ranks (with auto-calculated `2:1` Alpha formulation):

```bash
MODEL="google/gemma-4-E2B-it" bash runs/sweep_pipeline.sh
```

*(You can edit `runs/sweep_pipeline.sh` manually to alter the loop iterations).*

---

## 5. Submitting to an HPC Cluster (SLURM)

To allocate A10/A100 nodes via the SLURM workload manager without disrupting your active environments, use our wrapper file:
```bash
MODEL="google/gemma-4-E2B-it" sbatch runs/slurm_full_pipeline.sbatch
```
You can pass standard flags directly before `sbatch` exactly as you do locally.

---

## 6. Helpful Toggles / Quick Sanity Checks

To quickly debug the model end-to-end without waiting for massive datasets:

```bash
MODEL="google/gemma-4-E2B-it" MAX_TRAIN_SAMPLES=100 MAX_EVAL_SAMPLES=50 EPOCHS=1 bash runs/full_pipeline.sh
```

**Common Feature Flags:**
* **Disable Flash Attention completely** and revert to PyTorch's standard `sdpa`: `NO_FLASH_ATTN=1`
* **Disable Species-Specific filtering**: `SPECIES_SPECIFIC_OPTIONS=0`
* **Run Post-Training predictions** (generates `eval_predictions.csv`): `RUN_EVAL_INFERENCE=1` 
* **Alter LoRA architecture** (Example): `LORA_R=64 LORA_ALPHA=128 LORA_DROPOUT=0.05`
* **Start local web server** instantly after training finishes: `START_WEB=1`# Gemma 4 Fine-Tuning Runbook

This runbook describes how to execute the SFT pipeline specifically tailored for tuning `google/gemma-4-E2B-it` via our orchestrator script `runs/full_pipeline.sh`.

## 1. Fresh Start (Standard Run)

To start a fresh training run for Gemma 4 using the default dataset (`plant_village_classification`):

```bash
MODEL="google/gemma-4-E2B-it" bash runs/full_pipeline.sh
```

**What this does:**
1. Automatically spins up/syncs `uv` virtual environments based on `pyproject.toml`.
2. Resolves and reinstalls accurate `torch`/`torchvision` bindings for your host GPU out-of-band.
3. Dynamically identifies your CUDA/PyTorch combination and installs the fastest *pre-built* **Flash Attention 2** wheel (no slow builds!).
4. Prepares the AgML dataset (e.g. enforcing species-specific options dynamically).
5. Tunes the model using standard parameters (LoRA R=16, Alpha=32).

---

## 2. Fast Development Rerun

If your environment (`.venv`) is already fully compiled and ready, and you just want to rerun training extremely quickly:

```bash
MODEL="google/gemma-4-E2B-it" SKIP_ENV_SETUP=1 bash runs/full_pipeline.sh
```

---

## 3. Dealing with Crashes (OOM / Soft Resume)

If a training job was preempted, crashed, or encountered PyTorch's infamous `optimizer.pt` bloat/mis-dimension error on resume, use the `SOFT_RESUME=1` environment variable.

Just match your exact param flags, provide the original `RUN_TAG`, and skip the env synchronization:

```bash
RUN_TAG="20260414_030352" \
MODEL="google/gemma-4-E2B-it" \
SKIP_ENV_SETUP=1 \
SOFT_RESUME=1 \
bash runs/full_pipeline.sh
```
> *Note: `SOFT_RESUME=1` safely isolates `optimizer.pt` before the Hugging Face Trainer boots it up, permitting dimension recovery and avoiding VRAM death spikes.*

## 4. Hyperparameter Sweeps

To execute a matrix grid search automatically iterating through varying combinations of Learning Rates and LoRA Ranks (with auto-calculated `2:1` Alpha formulation):

```bash
MODEL="google/gemma-4-E2B-it" bash runs/sweep_pipeline.sh
```

---

## 5. Submitting to an HPC Cluster (SLURM)

To allocate A10/A100 nodes via the SLURM workload manager without disrupting your active environments, use our wrapper file:
```bash
MODEL="google/gemma-4-E2B-it" sbatch runs/slurm_full_pipeline.sbatch
```

---

## 6. Helpful Toggles / Quick Sanity Checks

To quickly debug the model end-to-end without waiting for massive datasets:
```bash
MODEL="google/gemma-4-E2B-it" MAX_TRAIN_SAMPLES=100 MAX_EVAL_SAMPLES=50 EPOCHS=1 bash runs/full_pipeline.sh
```

**Common Feature Flags:**
* Disable Flash Attention completely: `NO_FLASH_ATTN=1`
* Disable Species-Specific filtering: `SPECIES_SPECIFIC_OPTIONS=0`
* Run Post-Training predictions (generates `eval_predictions.csv`): `RUN_EVAL_INFERENCE=1`
* Start local web server instantly after training finishes: `START_WEB=1`
