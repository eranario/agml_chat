from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agml_chat.modeling import _looks_like_legacy_lora_full_checkpoint, _resolve_base_model_for_adapter


def test_resolve_base_model_non_adapter_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "plain_model"
    model_dir.mkdir()
    base, adapter = _resolve_base_model_for_adapter(str(model_dir))
    assert base == str(model_dir)
    assert adapter is None


def test_resolve_base_model_from_adapter_config(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct"}',
        encoding="utf-8",
    )

    base, adapter = _resolve_base_model_for_adapter(str(adapter_dir))
    assert base == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert adapter == str(adapter_dir)


def test_resolve_adapter_missing_base_model_raises(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "broken_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        _resolve_base_model_for_adapter(str(adapter_dir))


def test_resolve_remote_adapter_via_peft_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_from_pretrained(name: str) -> SimpleNamespace:
        assert name == "eranario/agml-chat-lora"
        return SimpleNamespace(base_model_name_or_path="google/gemma-3-4b-it")

    monkeypatch.setattr("agml_chat.modeling.PeftConfig.from_pretrained", _fake_from_pretrained)

    base, adapter = _resolve_base_model_for_adapter("eranario/agml-chat-lora")
    assert base == "google/gemma-3-4b-it"
    assert adapter == "eranario/agml-chat-lora"


def test_detect_legacy_lora_full_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "legacy"
    model_dir.mkdir()
    payload = {
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": "model-00001.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.base_layer.weight": "model-00001.safetensors",
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")

    assert _looks_like_legacy_lora_full_checkpoint(model_dir)


def test_resolve_legacy_lora_full_checkpoint_raises(tmp_path: Path) -> None:
    model_dir = tmp_path / "legacy"
    model_dir.mkdir()
    payload = {
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": "model-00001.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.base_layer.weight": "model-00001.safetensors",
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy checkpoint format"):
        _resolve_base_model_for_adapter(str(model_dir))


def test_detect_legacy_lora_full_checkpoint_from_safetensors_only(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    import torch

    model_dir = tmp_path / "legacy_no_index"
    model_dir.mkdir()
    safetensors.save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros((2, 2)),
            "model.language_model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros((2, 2)),
        },
        str(model_dir / "model.safetensors"),
    )

    assert _looks_like_legacy_lora_full_checkpoint(model_dir)
