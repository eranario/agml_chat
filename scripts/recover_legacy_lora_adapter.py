#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path


def _is_adapter_tensor_key(key: str) -> bool:
    markers = (
        ".lora_A.default.",
        ".lora_B.default.",
        ".lora_embedding_A.default.",
        ".lora_embedding_B.default.",
        ".lora_magnitude_vector.default.",
    )
    return any(marker in key for marker in markers)


def _to_peft_adapter_key(raw_key: str) -> str:
    key = raw_key.replace(".default.", ".")
    if not key.startswith("base_model.model."):
        key = f"base_model.model.{key}"
    return key


def _collect_adapter_tensors(legacy_dir: Path) -> tuple[dict[str, object], set[str], set[int]]:
    try:
        safe_open = getattr(importlib.import_module("safetensors"), "safe_open")
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for legacy adapter recovery. "
            "Install with: pip install safetensors"
        ) from exc

    tensors: dict[str, object] = {}
    target_modules: set[str] = set()
    inferred_r_values: set[int] = set()

    safetensor_files = sorted(legacy_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {legacy_dir}")

    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt") as handle:
            for key in handle.keys():
                if not _is_adapter_tensor_key(key):
                    continue

                tensor = handle.get_tensor(key)
                peft_key = _to_peft_adapter_key(key)
                tensors[peft_key] = tensor

                parts = key.split(".")
                for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
                    if marker in parts:
                        idx = parts.index(marker)
                        if idx > 0:
                            target_modules.add(parts[idx - 1])
                        break

                if ".lora_A.default." in key and tensor.ndim >= 1:
                    inferred_r_values.add(int(tensor.shape[0]))
                elif ".lora_B.default." in key and tensor.ndim >= 2:
                    inferred_r_values.add(int(tensor.shape[-1]))

    if not tensors:
        raise ValueError(
            "No LoRA adapter tensors were found in legacy safetensors files. "
            "Expected keys containing .lora_A.default. or .lora_B.default."
        )

    return tensors, target_modules, inferred_r_values


def _write_adapter_config(
    output_dir: Path,
    base_model: str,
    target_modules: set[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    adapter_config = {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": sorted(target_modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_preprocessing_files(legacy_dir: Path, output_dir: Path) -> None:
    file_names = [
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
    ]
    for name in file_names:
        src = legacy_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    try:
        save_file = getattr(importlib.import_module("safetensors.torch"), "save_file")
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for legacy adapter recovery. "
            "Install with: pip install safetensors"
        ) from exc

    parser = argparse.ArgumentParser(
        description="Recover a PEFT LoRA adapter folder from a malformed legacy final folder."
    )
    parser.add_argument(
        "--legacy-model-dir",
        type=str,
        required=True,
        help="Path to malformed legacy final dir containing safetensors with lora_* keys.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output dir for recovered adapter package.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name/path (e.g. google/gemma-4-E2B-it).",
    )
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it exists.")
    args = parser.parse_args()

    legacy_dir = Path(args.legacy_model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not legacy_dir.exists() or not legacy_dir.is_dir():
        raise ValueError(f"Legacy model directory not found: {legacy_dir}")

    if output_dir.exists():
        if not args.force:
            raise ValueError(f"Output directory already exists: {output_dir}. Pass --force to overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors, target_modules, inferred_r_values = _collect_adapter_tensors(legacy_dir)

    if args.lora_r is not None:
        lora_r = args.lora_r
    else:
        if len(inferred_r_values) != 1:
            raise ValueError(
                "Could not infer a unique LoRA rank from checkpoint tensors. "
                f"Found ranks: {sorted(inferred_r_values)}. Pass --lora-r explicitly."
            )
        lora_r = next(iter(inferred_r_values))

    save_file(tensors, str(output_dir / "adapter_model.safetensors"))
    _write_adapter_config(
        output_dir=output_dir,
        base_model=args.base_model,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    _copy_preprocessing_files(legacy_dir=legacy_dir, output_dir=output_dir)

    print("Recovered adapter package created.")
    print(f"Legacy source: {legacy_dir}")
    print(f"Recovered output: {output_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Inferred/selected LoRA rank: {lora_r}")
    print(f"Target modules: {', '.join(sorted(target_modules))}")
    print("Run chat with --model pointing to the recovered output directory.")


if __name__ == "__main__":
    main()
