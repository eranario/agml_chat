from __future__ import annotations

from agml_chat.dataset import apply_chat_template, prepare_messages_for_template


class DummyProcessor:
    pass


def test_prepare_messages_injects_image_path_once() -> None:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": "ok"},
    ]
    out = prepare_messages_for_template(messages, "/tmp/x.png")
    assert out[0]["content"][0]["image"] == "/tmp/x.png"


def test_apply_chat_template_fallback() -> None:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is this?"}]},
    ]
    rendered = apply_chat_template(DummyProcessor(), messages, add_generation_prompt=True)
    assert "USER:" in rendered
    assert "ASSISTANT:" in rendered
