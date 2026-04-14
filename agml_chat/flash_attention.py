from __future__ import annotations

import importlib.util
import logging

import torch


from typing import Any

def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attention_implementation(device: str, use_flash_attention: bool, config: Any = None) -> str:
    if device != "cuda":
        return "eager"

    if not use_flash_attention:
        return "sdpa"

    if has_flash_attn():
        if config is not None:
            # Fallback to computing from hidden_size and num_attention_heads if head_dim is missing
            head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1))
            # Flash Attention 2 strictly only supports up to head dim 256
            if head_dim > 256:
                logging.warning(
                    f"Model head dimension ({head_dim}) > 256, which FlashAttention 2 does not support. "
                    "Automatically falling back to SDPA."
                )
                return "sdpa"
                
        return "flash_attention_2"

    logging.warning(
        "flash-attn package not installed; falling back to SDPA. "
        "Install flash-attn to use Flash Attention."
    )
    return "sdpa"


def resolve_device_map(device: str) -> str | dict[str, int] | None:
    if device == "cuda":
        return "auto"
    if device in {"cpu", "mps"}:
        return {"": 0} if device == "mps" else None
    return None


def torch_compile_safe(model: torch.nn.Module, enable: bool) -> torch.nn.Module:
    if not enable:
        return model
    try:
        return torch.compile(model)
    except Exception:  # pragma: no cover - compile support depends on runtime
        logging.warning("torch.compile failed; running without compilation.")
        return model
