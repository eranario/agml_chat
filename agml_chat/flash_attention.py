from __future__ import annotations

import importlib.util
import logging

import torch


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attention_implementation(device: str, use_flash_attention: bool) -> str:
    if device != "cuda":
        return "eager"

    if not use_flash_attention:
        return "sdpa"

    if has_flash_attn():
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
