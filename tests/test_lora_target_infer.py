from __future__ import annotations

import pytest
import torch

from agml_chat.modeling import _infer_lora_target_modules, maybe_wrap_lora


class PreferredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)


class FallbackModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.custom_linear = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 4)


def test_infer_lora_targets_prefers_attention_mlp_names() -> None:
    targets = _infer_lora_target_modules(PreferredModel())
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_infer_lora_targets_falls_back_to_linear_leaf_names() -> None:
    targets = _infer_lora_target_modules(FallbackModel())
    assert "custom_linear" in targets
    assert "lm_head" not in targets


def test_maybe_wrap_lora_falls_back_when_target_module_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PreferredModel()

    def fake_get_peft_model(_model, _config):
        raise ValueError(
            "Target module Gemma4ClippableLinear("
            "(linear): Linear(in_features=768, out_features=768, bias=False)"
            ") is not supported."
        )

    monkeypatch.setattr("agml_chat.modeling.get_peft_model", fake_get_peft_model)

    wrapped = maybe_wrap_lora(
        model=model,
        enabled=True,
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj"],
    )
    assert wrapped is model
