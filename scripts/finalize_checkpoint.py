#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer


def _read_base_model_from_adapter_config(checkpoint_dir: Path) -> str | None:
    adapter_cfg = checkpoint_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        return None
    data = json.loads(adapter_cfg.read_text(encoding="utf-8"))
    value = data.get("base_model_name_or_path")
    if not value:
        return None
    return str(value)


def _resolve_output_dir(checkpoint_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    parent = checkpoint_dir.parent
    return (parent / "final").resolve()


def _materialize_checkpoint(checkpoint_dir: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise ValueError(
                f"Output directory already exists: {output_dir}. "
                "Pass --force to overwrite it."
            )
        shutil.rmtree(output_dir)

    shutil.copytree(checkpoint_dir, output_dir)


def _save_preprocessing_artifacts(base_model: str, output_dir: Path, trust_remote_code: bool) -> None:
    # Save any available processor/tokenizer/image configs to maximize compatibility
    # with model-specific Auto* loading paths.
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    processor.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

    try:
        image_processor = AutoImageProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        image_processor.save_pretrained(output_dir)
    except Exception:
        pass

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        feature_extractor.save_pretrained(output_dir)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a final inference folder from an existing checkpoint without training."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to a saved checkpoint directory (e.g. runs/sft_x/checkpoint-5700)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output final folder. Defaults to sibling 'final' next to checkpoint dir.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base model id/path for processor fallback (e.g. google/gemma-4-E2B-it).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading processor.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    output_dir = _resolve_output_dir(checkpoint_dir=checkpoint_dir, output_dir=args.output_dir)
    base_model = args.base_model or _read_base_model_from_adapter_config(checkpoint_dir)
    if not base_model:
        raise ValueError(
            "Could not infer base model from adapter_config.json. "
            "Pass --base-model explicitly."
        )

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Base model for processor: {base_model}")
    print(f"Output final dir: {output_dir}")

    _materialize_checkpoint(checkpoint_dir=checkpoint_dir, output_dir=output_dir, force=args.force)

    _save_preprocessing_artifacts(
        base_model=base_model,
        output_dir=output_dir,
        trust_remote_code=args.trust_remote_code,
    )

    preprocessor_cfg = output_dir / "preprocessor_config.json"
    if not preprocessor_cfg.exists():
        print(
            "Warning: preprocessor_config.json was not created. "
            "If chat loading fails, verify the base model id and transformers version."
        )

    print("Finalized checkpoint for inference.")
    print(f"Use this with CLI/Web --model: {output_dir}")


if __name__ == "__main__":
    main()
