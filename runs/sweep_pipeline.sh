#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Starting Sequential Hyperparameter Sweep"
echo "=========================================================="

# --- Fixed Configuration ---
export MODEL="google/gemma-4-E2B-it"
export NO_FLASH_ATTN=1
export RUN_EVAL_INFERENCE=1
export EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=1

# Optional: Cap max samples if you want the sweep to finish faster for debugging
# export MAX_TRAIN_SAMPLES=500
# export MAX_EVAL_SAMPLES=100

# --- Hyperparameter Grids ---
LEARNING_RATES=(1e-5 2e-5)
LORA_RANKS=(16 32)

total_runs=$((${#LEARNING_RATES[@]} * ${#LORA_RANKS[@]}))
run_count=1

for lr in "${LEARNING_RATES[@]}"; do
  for r in "${LORA_RANKS[@]}"; do
    alpha=$(( r * 2 ))
    echo "=========================================================="
    echo " Sweep Run $run_count / $total_runs"
    echo " Parameters: LEARNING_RATE=$lr | LORA_R=$r | LORA_ALPHA=$alpha"
    echo "=========================================================="

    # Export the sweep variables
    export LEARNING_RATE="$lr"
    export LORA_R="$r"
    export LORA_ALPHA="$alpha"

    # Run the pipeline (generates a new timestamped folder per run)
    bash runs/full_pipeline.sh

    echo "=========================================================="
    echo " Finished run $run_count"
    echo "=========================================================="
    
    run_count=$((run_count + 1))
    
    # Optional cool-down period between runs to let GPU memory fully release
    sleep 5
  done
done

echo "All sweep runs completed successfully!"