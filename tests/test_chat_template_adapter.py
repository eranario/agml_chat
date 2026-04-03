from __future__ import annotations

from agml_chat.chat_template_adapter import (
    ModelFamily,
    detect_model_family,
    normalize_messages_for_family,
)


def test_detect_model_family_qwen_25_vl() -> None:
    family = detect_model_family("Qwen/Qwen2.5-VL-3B-Instruct")
    assert family == ModelFamily.QWEN_VL


def test_detect_model_family_qwen_3_vl() -> None:
    family = detect_model_family("Qwen/Qwen3-VL-8B-Instruct")
    assert family == ModelFamily.QWEN_VL


def test_detect_model_family_generic_non_qwen() -> None:
    family = detect_model_family("meta-llama/Llama-3.2-11B-Vision-Instruct")
    assert family == ModelFamily.GENERIC


def test_normalize_qwen_messages_converts_all_turns_to_typed_blocks() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this crop."}]},
        {"role": "assistant", "content": "healthy"},
    ]

    normalized = normalize_messages_for_family(
        messages=messages,
        family=ModelFamily.QWEN_VL,
        image_path="/tmp/leaf.png",
    )

    assert isinstance(normalized[0]["content"], list)
    assert normalized[0]["content"][0] == {"type": "text", "text": "You are helpful."}
    assert normalized[1]["content"][0]["type"] == "image"
    assert normalized[1]["content"][0]["image"] == "/tmp/leaf.png"
    assert normalized[2]["content"][0] == {"type": "text", "text": "healthy"}


def test_normalize_generic_messages_preserves_string_content() -> None:
    messages = [
        {"role": "system", "content": "Keep as string"},
        {"role": "assistant", "content": "Also string"},
    ]
    normalized = normalize_messages_for_family(messages=messages, family=ModelFamily.GENERIC)
    assert normalized[0]["content"] == "Keep as string"
    assert normalized[1]["content"] == "Also string"
