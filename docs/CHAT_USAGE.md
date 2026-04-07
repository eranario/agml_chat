# Chat Usage Guide (CLI and Web UI)

This guide explains how to run chat inference in two modes:

- CLI chat for lower overhead and easier memory control.
- Web chat UI for browser-based interaction.

## Prerequisites

From the repository root:

```bash
uv sync --extra cpu
# or, if you have supported GPU setup:
# uv sync --extra gpu
source .venv/bin/activate
```

## Option A: CLI Chat (Recommended for constrained hardware)

Run interactive chat:

```bash
uv run -m scripts.chat_cli \
  --model runs/sft_gemma4_e2b_it/final \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 128
```

### Why CLI is usually lighter

- No web server process.
- No browser tab overhead.
- Easier to stop cleanly and release memory.

### Useful CLI flags

- `--device`: `auto`, `cuda`, `cpu`, or `mps`.
- `--dtype`: `float16`, `bfloat16`, or `float32`.
- `--max-new-tokens`: lowers generation length and memory pressure.
- `--temperature`, `--top-p`: sampling controls.
- `--prompt-config`: custom prompt template YAML.
- `--enable-thinking`: enables Gemma 4 thinking mode when supported.
- `--no-flash-attn`: disable flash attention path.

### CLI commands at runtime

- `/research on` and `/research off`
- `/image /path/to/image.jpg`
- `/clear` clears conversation history (does not unload model)
- `/quit` exits and releases model memory

### One-shot mode

For short jobs that should exit immediately after one answer:

```bash
uv run -m scripts.chat_cli \
  --model runs/sft_gemma4_e2b_it/final \
  --device cpu \
  --single-prompt "Identify likely disease symptoms from this image." \
  --image /path/to/leaf.jpg
```

## Option B: Web Chat UI

Start the server:

```bash
uv run -m scripts.chat_web \
  --model runs/sft_gemma4_e2b_it/final \
  --host 0.0.0.0 \
  --port 8000
```

Open in browser:

- http://localhost:8000

### Web mode notes

- This mode keeps model memory allocated while server is running.
- Web mode adds FastAPI plus browser overhead compared with CLI.
- Stop server with Ctrl+C to release memory.

## Memory and process control tips

- Prefer Ctrl+C or `/quit` to stop chat cleanly.
- Avoid Ctrl+Z for model runs. It suspends the process, and memory usually remains allocated.
- If you used Ctrl+Z accidentally:

```bash
jobs
fg %1
# then exit with /quit or Ctrl+C
```

Or terminate the suspended job directly:

```bash
kill %1
```

## Suggested low-memory defaults

- Use `--device cpu` for stability on limited unified memory.
- Start with `--max-new-tokens 64` to `128`.
- Keep one active chat process at a time.
- Use smaller model checkpoints where possible.

## Checkpoint Recovery (No More Training)

If training stopped before `runs/.../final` was created, use this exact sequence to create a runnable final folder from a checkpoint and launch CLI.

```bash
cd /group/jmearlesgrp/scratch/eranario/agml_chat
git pull
source .venv/bin/activate

# Gemma 4 compatibility: make sure Transformers is recent enough.
uv pip install --python .venv/bin/python --upgrade "git+https://github.com/huggingface/transformers.git"

# Materialize a final folder from checkpoint (no training).
python -m scripts.finalize_checkpoint \
  --checkpoint-dir runs/sft_20260406_130349/checkpoint-5700 \
  --output-dir runs/sft_20260406_130349/final \
  --base-model google/gemma-4-E2B-it \
  --trust-remote-code \
  --force

# Verify final folder has model + processor/tokenizer files.
ls -lah runs/sft_20260406_130349/final
ls -lah runs/sft_20260406_130349/final/preprocessor_config.json

# Run CLI with absolute model path.
python -m scripts.chat_cli \
  --model /output/final \
  --device cuda \
  --dtype float32 \
  --max-new-tokens 128
```

Note: after a manual Transformers source upgrade, prefer `python -m ...` from the activated `.venv`. `uv run` may reconcile packages and roll Transformers back to the lockfile version.

If `finalize_checkpoint` fails, inspect checkpoint contents:

```bash
ls -lah runs/sft_20260406_130349/checkpoint-5700
```

If you prefer to continue training from a checkpoint instead of finalizing it for inference, use the resume commands in `docs/LAMBDA_SCRIPT_RUNBOOK.md` (section `3e`).

```
python -m scripts.chat_sft \
--model-name google/gemma-4-E2B-it \
--train-jsonl data/agml_sft_20260406_130349/train.jsonl \
--val-jsonl data/agml_sft_20260406_130349/val.jsonl \
--output-dir runs/sft_20260406_130349_resume2 \
--resume-from-checkpoint runs/sft_20260406_130349/checkpoint-6240 \
--max-train-samples 10 \
--max-eval-samples 1 \
--device cuda \
--dtype bfloat16 \
--epochs 1 \
--per-device-train-batch-size 1 \
--gradient-accumulation-steps 8 \
--save-steps 20 \
--logging-steps 5 \
--live-metrics 
```