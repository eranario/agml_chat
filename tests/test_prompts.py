from __future__ import annotations

from pathlib import Path

import pytest

from agml_chat.prompts import load_prompt_set


def test_load_prompt_set_defaults() -> None:
    prompt_set = load_prompt_set()
    assert "agricultural" in prompt_set.system_prompt.lower()
    assert "diagnosis" in prompt_set.classification_instruction.lower()
    assert "diagnosis" in prompt_set.inference_instruction.lower()


def test_load_prompt_set_rejects_unknown_key(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("unknown_key: value\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_prompt_set(str(cfg))
