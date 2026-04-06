from __future__ import annotations

from pathlib import Path

from agml_chat.modeling import _resolve_base_model_for_adapter


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
