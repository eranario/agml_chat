from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from agml_chat.chat_template_adapter import ModelFamily, detect_model_family
from agml_chat.common import RuntimeConfig
from agml_chat.flash_attention import resolve_attention_implementation



def _infer_lora_target_modules(model: torch.nn.Module) -> list[str]:
    """Infer LoRA target module names from Linear layer leaf names."""
    preferred = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv_proj",
    )
    excluded = {"lm_head", "output", "classifier", "score"}

    linear_leaf_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_leaf_names.add(name.rsplit(".", 1)[-1])
        elif getattr(module, "linear", None) is not None and isinstance(module.linear, torch.nn.Linear):
            # Gracefully handle custom wrappers like Gemma4ClippableLinear
            base_name = name.rsplit(".", 1)[-1]
            linear_leaf_names.add(f"{base_name}.linear")

    inferred = [name for name in preferred if name in linear_leaf_names or f"{name}.linear" in linear_leaf_names]
    # For custom wrappers, if the base name was preferred, ensure we target the `.linear` leaf
    final_inferred = []
    for pref in inferred:
        if f"{pref}.linear" in linear_leaf_names:
            final_inferred.append(f"{pref}.linear")
        else:
            final_inferred.append(pref)

    if final_inferred:
        return final_inferred

    fallback = sorted(name for name in linear_leaf_names if name not in excluded)
    if fallback:
        return fallback

    raise ValueError(
        "Could not infer LoRA target modules from model architecture. "
        "Pass --lora-target-modules explicitly (comma-separated)."
    )


def _try_model_loader(loader: Any, model_name: str, kwargs: dict) -> torch.nn.Module | None:
    if loader is None:
        return None
    try:
        return loader.from_pretrained(model_name, **kwargs)
    except Exception:
        return None


def _looks_like_legacy_lora_full_checkpoint(path: Path) -> bool:
    """Detect full-model checkpoints that accidentally contain PEFT LoRA keys."""
    key_markers = (".lora_A.", ".lora_B.", ".base_layer.")

    index_candidates = [
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ]
    for index_path in index_candidates:
        if not index_path.exists():
            continue
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map", {})
            if not isinstance(weight_map, dict):
                continue
            for key in weight_map:
                if any(marker in key for marker in key_markers):
                    return True
        except Exception:
            continue

    # Some malformed exports omit index files and only ship one or more .safetensors shards.
    # Inspect tensor names from safetensors headers without materializing tensor data.
    try:
        from safetensors import safe_open
    except Exception:
        return False

    for tensor_file in sorted(path.glob("*.safetensors")):
        try:
            with safe_open(tensor_file, framework="pt") as handle:
                for key in handle.keys():
                    if any(marker in key for marker in key_markers):
                        return True
        except Exception:
            continue

    return False


def _resolve_base_model_for_adapter(model_name: str, token: str | None = None) -> tuple[str, str | None]:
    """Return (base_model_name, adapter_path) when model_name points to a PEFT adapter."""
    path = Path(model_name)
    adapter_config_path = path / "adapter_config.json"

    # Fast-path: local adapter directories with adapter_config.json present.
    if path.exists() and path.is_dir() and adapter_config_path.exists():
        try:
            peft_cfg = PeftConfig.from_pretrained(str(path), token=token)
        except Exception as exc:
            raise ValueError(f"Failed to parse adapter config at {adapter_config_path}: {exc}") from exc

        base_model_name = getattr(peft_cfg, "base_model_name_or_path", None)
        if not base_model_name:
            raise ValueError(
                f"Adapter config at {adapter_config_path} is missing 'base_model_name_or_path'."
            )
        return str(base_model_name), str(path)

    # Fail fast on legacy checkpoints that contain LoRA keys in full-model shards.
    if path.exists() and path.is_dir() and _looks_like_legacy_lora_full_checkpoint(path):
        raise ValueError(
            "Detected a legacy checkpoint format: full-model shard files contain LoRA keys, "
            "but adapter_config.json is missing. This cannot be loaded reliably as a base model. "
            "Use a proper PEFT adapter folder (adapter_config.json + adapter_model.safetensors) "
            "or re-finalize from a checkpoint that includes adapter artifacts."
        )

    # Fallback: handle adapter repos on the Hub (or cached adapters) by probing PEFT config.
    try:
        peft_cfg = PeftConfig.from_pretrained(model_name, token=token)
    except Exception:
        return model_name, None

    base_model_name = getattr(peft_cfg, "base_model_name_or_path", None)
    if not base_model_name:
        raise ValueError(
            f"PEFT config for '{model_name}' is missing 'base_model_name_or_path'."
        )
    return str(base_model_name), model_name



def load_model_and_processor(
    model_name: str,
    runtime: RuntimeConfig,
    use_flash_attention: bool,
    trust_remote_code: bool = True,
    token: str | None = None,
) -> tuple[torch.nn.Module, Any, ModelFamily]:
    base_model_name, adapter_path = _resolve_base_model_for_adapter(model_name, token=token)
    logging.info(
        "Model resolution: input=%s base=%s adapter=%s",
        model_name,
        base_model_name,
        adapter_path if adapter_path is not None else "<none>",
    )

    processor = None
    processor_sources = [model_name]
    if adapter_path is not None and base_model_name != model_name:
        processor_sources.append(base_model_name)
    last_processor_exc: Exception | None = None
    for source in processor_sources:
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code, token=token)
            break
        except Exception as exc:
            last_processor_exc = exc
    if processor is None:
        assert last_processor_exc is not None
        raise last_processor_exc

    try:
        config = AutoConfig.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code, token=token
        )
    except Exception:
        config = None

    if config is None and "gemma" in base_model_name.lower():
        logging.warning("Gemma model detected but config failed to load. Forcing SDPA to prevent FlashAttention 2 head dimension overflow.")
        use_flash_attention = False

    attention_impl = resolve_attention_implementation(runtime.device, use_flash_attention, config=config)
    model_kwargs = {
        "torch_dtype": runtime.torch_dtype,
        "trust_remote_code": trust_remote_code,
        "token": token,
    }

    # Attention implementation is optional for some model classes.
    model_kwargs_with_attn = {**model_kwargs, "attn_implementation": attention_impl}

    # Load with the broadest VLM classes available, fallback to CausalLM.
    image_text_loader = getattr(transformers, "AutoModelForImageTextToText", None)
    vision2seq_loader = getattr(transformers, "AutoModelForVision2Seq", None)

    model = _try_model_loader(image_text_loader, base_model_name, model_kwargs_with_attn)
    if model is None:
        model = _try_model_loader(vision2seq_loader, base_model_name, model_kwargs_with_attn)
    if model is None:
        model = _try_model_loader(AutoModelForCausalLM, base_model_name, model_kwargs_with_attn)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, token=token)

    if runtime.device == "cuda":
        model = model.to("cuda")
    elif runtime.device == "mps":
        model = model.to("mps")
    else:
        model = model.to("cpu")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_family = detect_model_family(model_name_or_path=base_model_name, processor=processor, model=model)
    logging.info("Loaded model %s with attention implementation '%s'", base_model_name, attention_impl)
    if adapter_path is not None:
        logging.info("Loaded PEFT adapter from %s", adapter_path)
    logging.info("Detected model family: %s", model_family.value)
    return model, processor, model_family



def maybe_wrap_lora(
    model: torch.nn.Module,
    enabled: bool,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str] | None,
) -> torch.nn.Module:
    if not enabled:
        return model
    if target_modules is None:
        target_modules = _infer_lora_target_modules(model)
        logging.info("Auto-inferred LoRA target modules: %s", ", ".join(target_modules))

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        use_rslora=False,
    )
    try:
        # Note: Do NOT explicitly unwrap custom Gemma 4 layer abstractions here.
        # Mutating the module tree removes `Gemma4ClippableLinear`, causing 
        # checkpoint saves to permanently lose the correct Hugging Face key paths 
        # (e.g. they save `o_proj.weight` instead of `o_proj.linear.weight`).
        # The auto-inferrer `_infer_lora_target_modules` now natively targets 
        # `q_proj.linear` natively, so PEFT never throws Unsupported Module errors!

        wrapped = get_peft_model(model, config)
    except ValueError as exc:
        message = str(exc)
        if "Target module" in message and "is not supported" in message:
            logging.warning(
                "LoRA could not wrap one or more target modules (%s). "
                "Continuing without LoRA for this run. "
                "Set --no-lora to disable this warning.",
                message,
            )
            return model
        raise
    wrapped.print_trainable_parameters()
    return wrapped
