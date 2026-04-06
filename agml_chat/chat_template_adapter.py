from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class ModelFamily(str, Enum):
    QWEN_VL = "qwen_vl"
    GEMMA_VL = "gemma_vl"
    GENERIC = "generic"


_GEMMA_4_MARKERS = (
    "gemma-4",
    "gemma4",
    "gemma_4",
)

_GEMMA_4_MODEL_NAMES = (
    "google/gemma-4-e2b-it",
    "google/gemma-4-e4b-it",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
)


def detect_model_family(
    model_name_or_path: str | None = None,
    processor: Any | None = None,
    model: Any | None = None,
) -> ModelFamily:
    """Best-effort model family detection for prompt/template normalization."""
    candidates: list[str] = []
    if model_name_or_path:
        candidates.append(str(model_name_or_path).lower())

    if processor is not None:
        candidates.append(processor.__class__.__name__.lower())
        proc_model_type = getattr(processor, "model_input_names", None)
        if proc_model_type is not None:
            candidates.append(str(proc_model_type).lower())

    if model is not None:
        candidates.append(model.__class__.__name__.lower())
        config = getattr(model, "config", None)
        if config is not None:
            for key in ("model_type",):
                value = getattr(config, key, None)
                if value:
                    candidates.append(str(value).lower())
            arch = getattr(config, "architectures", None)
            if arch:
                candidates.extend(str(item).lower() for item in arch)

    joined = " ".join(candidates)
    qwen_markers = (
        "qwen2.5-vl",
        "qwen3-vl",
        "qwen2_vl",
        "qwen3_vl",
    )
    if any(marker in joined for marker in qwen_markers):
        return ModelFamily.QWEN_VL
    if "qwen" in joined and "vl" in joined:
        return ModelFamily.QWEN_VL

    if any(marker in joined for marker in _GEMMA_4_MARKERS) or any(
        model_name in joined for model_name in _GEMMA_4_MODEL_NAMES
    ):
        return ModelFamily.GEMMA_VL
    return ModelFamily.GENERIC


def _normalize_generic_messages(messages: list[dict[str, Any]], image_path: str | None = None) -> list[dict[str, Any]]:
    """Preserve existing behavior and only inject image path when requested."""
    normalized: list[dict[str, Any]] = []
    image_injected = False

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            new_content: list[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if image_path and not image_injected and "image" not in item:
                        item = {**item, "image": image_path}
                        image_injected = True
                new_content.append(item)
            normalized.append({**message, "content": new_content})
        else:
            normalized.append({**message})

    return normalized


def _normalize_qwen_vl_messages(messages: list[dict[str, Any]], image_path: str | None = None) -> list[dict[str, Any]]:
    """Normalize all turns into typed content blocks expected by Qwen VL templates."""
    normalized: list[dict[str, Any]] = []
    image_injected = False

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        typed_content: list[dict[str, Any]] = []
        if isinstance(content, str):
            typed_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "image":
                        image_item = dict(item)
                        if image_path and not image_injected and "image" not in image_item:
                            image_item["image"] = image_path
                            image_injected = True
                        typed_content.append(image_item)
                    elif item_type == "text":
                        typed_content.append({"type": "text", "text": item.get("text", "")})
                    else:
                        # Keep unknown typed entries intact for forward compatibility.
                        typed_content.append(dict(item))
                elif isinstance(item, str):
                    typed_content.append({"type": "text", "text": item})
                else:
                    typed_content.append({"type": "text", "text": str(item)})
        else:
            typed_content.append({"type": "text", "text": str(content)})

        normalized.append({"role": role, "content": typed_content})

    return normalized


def _normalize_gemma_vl_messages(messages: list[dict[str, Any]], image_path: str | None = None) -> list[dict[str, Any]]:
    """Normalize turns into typed blocks compatible with Gemma 4 chat templates."""
    normalized: list[dict[str, Any]] = []
    image_injected = False

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        typed_content: list[dict[str, Any]] = []
        if isinstance(content, str):
            typed_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "image":
                        image_item = dict(item)
                        if image_path and not image_injected and "image" not in image_item and "url" not in image_item:
                            image_item["image"] = image_path
                            image_injected = True
                        typed_content.append(image_item)
                    elif item_type == "text":
                        typed_content.append({"type": "text", "text": item.get("text", "")})
                    else:
                        typed_content.append(dict(item))
                elif isinstance(item, str):
                    typed_content.append({"type": "text", "text": item})
                else:
                    typed_content.append({"type": "text", "text": str(item)})
        else:
            typed_content.append({"type": "text", "text": str(content)})

        normalized.append({"role": role, "content": typed_content})

    return normalized


MessageNormalizer = Callable[[list[dict[str, Any]], str | None], list[dict[str, Any]]]
_NORMALIZER_REGISTRY: dict[ModelFamily, MessageNormalizer] = {
    ModelFamily.QWEN_VL: _normalize_qwen_vl_messages,
    ModelFamily.GEMMA_VL: _normalize_gemma_vl_messages,
    ModelFamily.GENERIC: _normalize_generic_messages,
}


def normalize_messages_for_family(
    messages: list[dict[str, Any]],
    family: ModelFamily,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    normalizer = _NORMALIZER_REGISTRY.get(family, _normalize_generic_messages)
    return normalizer(messages, image_path=image_path)


def family_supports_thinking(family: ModelFamily) -> bool:
    return family == ModelFamily.GEMMA_VL


def apply_family_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    family: ModelFamily,
    add_generation_prompt: bool,
) -> str:
    if hasattr(processor, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if family == ModelFamily.GEMMA_VL:
            kwargs["enable_thinking"] = False
        return processor.apply_chat_template(messages, **kwargs)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if family == ModelFamily.GEMMA_VL:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)

    # Fallback format when no chat template exists.
    chunks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [part.get("text", "<image>") for part in content if isinstance(part, dict)]
            content = " ".join(text_parts)
        chunks.append(f"{role}: {content}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:")
    return "\n".join(chunks)
