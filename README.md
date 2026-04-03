# agml-chat

`agml-chat` is a nanochat-style scaffold (https://github.com/karpathy/nanochat/tree/master) for multimodal supervised fine-tuning on AgML image classification datasets.

It keeps a familiar `scripts/` entrypoint structure (`chat_sft.py`, `chat_cli.py`, `chat_web.py`) while replacing language-only data with image+text chat supervision.

## What This Implementation Covers

- Nanochat-like script workflow and chat interfaces
- Multimodal chat in CLI and web UI (image + text)
- Flash-attention aware model loading (with safe fallback)
- Research mode for multi-pass response generation
- AgML classification dataset selection and conversion
- Prompt template customization for training and inference
- Scaling knobs for model choice, LoRA, and training hyperparameters

## Project Layout

- `agml_chat/agml_data.py`: AgML dataset listing, loading, splitting, JSONL export
- `agml_chat/training.py`: HuggingFace Trainer + multimodal collator + LoRA hooks
- `agml_chat/engine.py`: inference engine + token streaming
- `agml_chat/web.py`: FastAPI app for chat endpoints
- `agml_chat/research.py`: research-mode multi-pass generation
- `agml_chat/prompts.py`: prompt template management
- `agml_chat/templates/ui.html`: browser chat UI
- `scripts/prepare_agml_sft.py`: create train/val/test JSONL from AgML datasets
- `scripts/chat_sft.py`: fine-tune a VLM
- `scripts/chat_cli.py`: local interactive chat
- `scripts/chat_web.py`: web server

## Setup

`agml-chat` uses `uv` for environment + dependency management, similar to nanochat.

```bash
cd /Users/admin-eranario/Documents/Code/agml_chat
# first time or after dependency edits:
uv lock
uv sync --extra gpu
# or CPU/MPS:
# uv sync --extra cpu
source .venv/bin/activate
```

For development tooling (pytest, ruff, notebooks):

```bash
uv sync --extra gpu --group dev
```

For reproducible installs in CI:

```bash
uv sync --frozen --extra gpu
```

## 1) List Available AgML Classification Datasets

```bash
uv run -m scripts.list_agml_datasets --min-images 500
```

## 2) Convert Selected AgML Datasets to Multimodal SFT Data

```bash
uv run -m scripts.prepare_agml_sft \
  --datasets plant_village_classification,rice_leaf_disease_classification \
  --output-dir data/agml_sft \
  --prompt-config configs/prompt_config.example.yaml
```

Generated files:

- `data/agml_sft/train.jsonl`
- `data/agml_sft/val.jsonl`
- `data/agml_sft/test.jsonl` (if configured)
- `data/agml_sft/dataset_manifest.json`

Each row includes:

- `image_path`
- `label`
- `labels`
- `messages` (system + user image/text + assistant answer)

## 3) Fine-tune a VLM

```bash
uv run -m scripts.chat_sft \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --train-jsonl data/agml_sft/train.jsonl \
  --val-jsonl data/agml_sft/val.jsonl \
  --output-dir runs/sft_qwen25vl3b \
  --epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

Notes:

- LoRA is enabled by default (`--no-lora` to disable)
- Flash attention path is enabled by default (`--no-flash-attn` to disable)
- Final model artifacts are saved to `runs/.../final`

## 4) Chat With the Model (CLI)

```bash
uv run -m scripts.chat_cli \
  --model runs/sft_qwen25vl3b/final \
  --image /path/to/leaf.jpg
```

CLI commands:

- `/research on` or `/research off`
- `/image /path/to/image.jpg`
- `/clear`
- `/quit`

## 5) Chat With the Model (Web UI)

```bash
uv run -m scripts.chat_web \
  --model runs/sft_qwen25vl3b/final \
  --host 0.0.0.0 \
  --port 8000
```

Open `http://localhost:8000`.

## Prompt Customization

Use `configs/prompt_config.example.yaml` as a template and pass it with `--prompt-config` in `prepare_agml_sft.py`, `chat_cli.py`, or `chat_web.py`.

Key configurable templates:

- `system_prompt`
- `classification_instruction`
- `research_mode_system_prompt`
- `inference_instruction`

## Qwen VL Compatibility

`agml-chat` includes a model-family chat-template adapter layer.

- Qwen2.5-VL and Qwen3-VL models are auto-detected and normalized into typed multimodal content blocks before `apply_chat_template`.
- Normalization is applied in both inference (`chat_cli.py`, `chat_web.py`) and SFT collation (`chat_sft.py` path), so prompt/history/image formatting stays consistent.
- Non-Qwen models use the generic adapter path, preserving existing behavior.
- The adapter registry is extensible for adding future model-family formatters without changing dataset export format.

## Runtime Safety (Nanochat-style)

The web chat server includes request validation and abuse-prevention limits:

- max messages per request
- max message length and total conversation length
- bounded generation params (`temperature`, `top_p`, `max_new_tokens`)
## Scaling Guidance

To scale across models and datasets:

1. Keep `prepare_agml_sft.py` output data model-agnostic and regenerate per dataset mix.
2. Sweep model IDs in `--model-name` without changing conversion logic.
3. Use LoRA target modules per architecture (`--lora-target-modules`) when needed.
4. Increase dataset breadth by adding more AgML dataset names to `--datasets`.
5. Adjust batch/accumulation for available GPU memory.

## Known Limitations

- Current AgML conversion targets image classification only.
- Evaluation metrics are not yet included in this first scaffold.
- Additional VLM families beyond Qwen may need family-specific adapter entries.

## Quick Start Script

- `runs/prepare_and_train_example.sh`: end-to-end dataset prep + SFT example
- `runs/runcpu.sh`: CPU smoke run for CLI multimodal inference
