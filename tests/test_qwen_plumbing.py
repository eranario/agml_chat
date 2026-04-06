from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def test_engine_generate_uses_family_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from agml_chat.chat_template_adapter import ModelFamily
    from agml_chat.common import RuntimeConfig
    from agml_chat.engine import ChatEngine

    calls: dict[str, Any] = {}

    def fake_normalize(messages, family, image_path=None):
        calls["normalize_family"] = family
        calls["normalize_image_path"] = image_path
        return messages

    def fake_apply_template(processor, messages, family, add_generation_prompt, enable_thinking=False):
        calls["template_family"] = family
        calls["template_add_generation_prompt"] = add_generation_prompt
        calls["template_enable_thinking"] = enable_thinking
        return "PROMPT"

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    class DummyProcessor:
        tokenizer = DummyTokenizer()

    class DummyModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr("agml_chat.engine.normalize_messages_for_family", fake_normalize)
    monkeypatch.setattr("agml_chat.engine.apply_family_chat_template", fake_apply_template)

    engine = ChatEngine(
        model=DummyModel(),
        processor=DummyProcessor(),
        runtime=RuntimeConfig(device="cpu", torch_dtype=torch.float32),
        model_family=ModelFamily.QWEN_VL,
    )

    monkeypatch.setattr(
        engine,
        "_prepare_inputs",
        lambda text, image_path: {"input_ids": torch.tensor([[1, 2]])},
    )

    result = engine.generate(prompt="what is this?", image_path="/tmp/image.png")
    assert result == "decoded"
    assert calls["normalize_family"] == ModelFamily.QWEN_VL
    assert calls["normalize_image_path"] == "/tmp/image.png"
    assert calls["template_family"] == ModelFamily.QWEN_VL
    assert calls["template_add_generation_prompt"] is True
    assert calls["template_enable_thinking"] is False


def test_engine_generate_passes_thinking_for_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from agml_chat.chat_template_adapter import ModelFamily
    from agml_chat.common import RuntimeConfig
    from agml_chat.engine import ChatEngine

    calls: dict[str, Any] = {}

    def fake_normalize(messages, family, image_path=None):
        calls["normalize_family"] = family
        return messages

    def fake_apply_template(processor, messages, family, add_generation_prompt, enable_thinking=False):
        calls["template_family"] = family
        calls["template_enable_thinking"] = enable_thinking
        return "PROMPT"

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    class DummyProcessor:
        tokenizer = DummyTokenizer()

    class DummyModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr("agml_chat.engine.normalize_messages_for_family", fake_normalize)
    monkeypatch.setattr("agml_chat.engine.apply_family_chat_template", fake_apply_template)

    engine = ChatEngine(
        model=DummyModel(),
        processor=DummyProcessor(),
        runtime=RuntimeConfig(device="cpu", torch_dtype=torch.float32),
        model_family=ModelFamily.GEMMA_VL,
    )

    monkeypatch.setattr(
        engine,
        "_prepare_inputs",
        lambda text, image_path: {"input_ids": torch.tensor([[1, 2]])},
    )

    result = engine.generate(prompt="what is this?", image_path="/tmp/image.png", enable_thinking=True)
    assert result == "decoded"
    assert calls["normalize_family"] == ModelFamily.GEMMA_VL
    assert calls["template_family"] == ModelFamily.GEMMA_VL
    assert calls["template_enable_thinking"] is True


def test_collator_uses_family_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from agml_chat.chat_template_adapter import ModelFamily
    from agml_chat.training import VisionLanguageSFTCollator

    calls: dict[str, Any] = {}

    def fake_normalize(messages, family, image_path=None):
        calls["normalize_family"] = family
        calls["normalize_image_path"] = image_path
        return messages

    def fake_apply_template(processor, messages, family, add_generation_prompt, enable_thinking=False):
        calls["template_family"] = family
        calls["template_add_generation_prompt"] = add_generation_prompt
        calls["template_enable_thinking"] = enable_thinking
        return "PROMPT"

    class DummyProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[1, 0, 2]])}

    monkeypatch.setattr("agml_chat.training.normalize_messages_for_family", fake_normalize)
    monkeypatch.setattr("agml_chat.training.apply_family_chat_template", fake_apply_template)
    monkeypatch.setattr("agml_chat.training.load_image", lambda image_path: object())

    collator = VisionLanguageSFTCollator(
        processor=DummyProcessor(),
        model_family=ModelFamily.QWEN_VL,
        max_length=128,
    )
    batch = collator(
        [
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image.png",
            }
        ]
    )

    assert batch["labels"].shape == batch["input_ids"].shape
    assert calls["normalize_family"] == ModelFamily.QWEN_VL
    assert calls["normalize_image_path"] == "/tmp/image.png"
    assert calls["template_family"] == ModelFamily.QWEN_VL
    assert calls["template_add_generation_prompt"] is False
    assert calls["template_enable_thinking"] is False


def test_collator_passes_thinking_for_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from agml_chat.chat_template_adapter import ModelFamily
    from agml_chat.training import VisionLanguageSFTCollator

    calls: dict[str, Any] = {}

    def fake_normalize(messages, family, image_path=None):
        calls["normalize_family"] = family
        return messages

    def fake_apply_template(processor, messages, family, add_generation_prompt, enable_thinking=False):
        calls["template_family"] = family
        calls["template_enable_thinking"] = enable_thinking
        return "PROMPT"

    class DummyProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[1, 0, 2]])}

    monkeypatch.setattr("agml_chat.training.normalize_messages_for_family", fake_normalize)
    monkeypatch.setattr("agml_chat.training.apply_family_chat_template", fake_apply_template)
    monkeypatch.setattr("agml_chat.training.load_image", lambda image_path: object())

    collator = VisionLanguageSFTCollator(
        processor=DummyProcessor(),
        model_family=ModelFamily.GEMMA_VL,
        max_length=128,
        enable_thinking=True,
    )
    batch = collator(
        [
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image.png",
            }
        ]
    )

    assert batch["labels"].shape == batch["input_ids"].shape
    assert calls["normalize_family"] == ModelFamily.GEMMA_VL
    assert calls["template_family"] == ModelFamily.GEMMA_VL
    assert calls["template_enable_thinking"] is True


def test_collator_formats_gemma_images_per_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from agml_chat.chat_template_adapter import ModelFamily
    from agml_chat.training import VisionLanguageSFTCollator

    calls: dict[str, Any] = {}

    def fake_normalize(messages, family, image_path=None):
        return messages

    def fake_apply_template(processor, messages, family, add_generation_prompt, enable_thinking=False):
        return "PROMPT"

    class DummyProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def __call__(self, **kwargs):
            calls["images"] = kwargs["images"]
            batch_size = len(kwargs["text"])
            return {"input_ids": torch.ones((batch_size, 3), dtype=torch.long)}

    monkeypatch.setattr("agml_chat.training.normalize_messages_for_family", fake_normalize)
    monkeypatch.setattr("agml_chat.training.apply_family_chat_template", fake_apply_template)
    monkeypatch.setattr("agml_chat.training.load_image", lambda image_path: object())

    collator = VisionLanguageSFTCollator(
        processor=DummyProcessor(),
        model_family=ModelFamily.GEMMA_VL,
        max_length=128,
    )
    batch = collator(
        [
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image1.png",
            },
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image2.png",
            },
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image3.png",
            },
            {
                "messages": [{"role": "user", "content": "hello"}],
                "image_path": "/tmp/image4.png",
            },
        ]
    )

    assert batch["labels"].shape == batch["input_ids"].shape
    assert len(calls["images"]) == 4
    assert all(isinstance(item, list) and len(item) == 1 for item in calls["images"])
