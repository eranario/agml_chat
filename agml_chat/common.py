from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    torch_dtype: torch.dtype


def autodetect_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype: str | None, device: str) -> torch.dtype:
    if dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[key]

    if device == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def build_runtime_config(device: str = "auto", dtype: str | None = None) -> RuntimeConfig:
    resolved_device = autodetect_device(device)
    resolved_dtype = resolve_torch_dtype(dtype, resolved_device)
    if resolved_device == "mps" and resolved_dtype == torch.float16:
        resolved_dtype = torch.float32
    return RuntimeConfig(device=resolved_device, torch_dtype=resolved_dtype)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
