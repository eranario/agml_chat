from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor

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

    inferred = [name for name in preferred if name in linear_leaf_names]
    if inferred:
        return inferred

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


def _resolve_base_model_for_adapter(model_name: str) -> tuple[str, str | None]:
    """Return (base_model_name, adapter_path) when model_name points to a local PEFT adapter dir."""
    path = Path(model_name)
    if not path.exists() or not path.is_dir():
        return model_name, None

    adapter_config_path = path / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_name, None

    try:
        adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse adapter config at {adapter_config_path}: {exc}") from exc

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"Adapter config at {adapter_config_path} is missing 'base_model_name_or_path'."
        )
    return str(base_model_name), str(path)



def load_model_and_processor(
    model_name: str,
    runtime: RuntimeConfig,
    use_flash_attention: bool,
    trust_remote_code: bool = True,
) -> tuple[torch.nn.Module, Any, ModelFamily]:
    base_model_name, adapter_path = _resolve_base_model_for_adapter(model_name)

    processor = None
    processor_sources = [model_name]
    if adapter_path is not None and base_model_name != model_name:
        processor_sources.append(base_model_name)
    last_processor_exc: Exception | None = None
    for source in processor_sources:
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code)
            break
        except Exception as exc:
            last_processor_exc = exc
    if processor is None:
        assert last_processor_exc is not None
        raise last_processor_exc

    attention_impl = resolve_attention_implementation(runtime.device, use_flash_attention)
    model_kwargs = {
        "torch_dtype": runtime.torch_dtype,
        "trust_remote_code": trust_remote_code,
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
        model = PeftModel.from_pretrained(model, adapter_path)

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
    )
    try:
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
