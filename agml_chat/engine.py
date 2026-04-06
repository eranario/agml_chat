from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from transformers import TextIteratorStreamer

from agml_chat.chat_template_adapter import (
    ModelFamily,
    apply_family_chat_template,
    family_supports_thinking,
    normalize_messages_for_family,
)
from agml_chat.common import RuntimeConfig, build_runtime_config
from agml_chat.dataset import load_image
from agml_chat.modeling import load_model_and_processor


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95


class ChatEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        runtime: RuntimeConfig,
        model_family: ModelFamily,
    ):
        self.model = model
        self.processor = processor
        self.runtime = runtime
        self.model_family = model_family

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str | None = None,
        use_flash_attention: bool = True,
        trust_remote_code: bool = True,
    ) -> "ChatEngine":
        runtime = build_runtime_config(device=device, dtype=dtype)
        model, processor, model_family = load_model_and_processor(
            model_name=model_name_or_path,
            runtime=runtime,
            use_flash_attention=use_flash_attention,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return cls(model=model, processor=processor, runtime=runtime, model_family=model_family)

    def _build_messages(
        self,
        prompt: str,
        image_path: str | None,
        history: Iterable[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(list(history))

        content: list[dict[str, str]] | str
        if image_path:
            content = [{"type": "image"}, {"type": "text", "text": prompt}]
        else:
            content = prompt
        messages.append({"role": "user", "content": content})

        return messages

    def _prepare_inputs(self, text: str, image_path: str | None) -> dict[str, torch.Tensor]:
        image = load_image(image_path) if image_path else None
        kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if image is not None:
            kwargs["images"] = [image]

        inputs = self.processor(**kwargs)
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        history: Iterable[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        generation: GenerationConfig | None = None,
        enable_thinking: bool = False,
    ) -> str:
        generation = generation or GenerationConfig()
        messages = self._build_messages(
            prompt=prompt,
            image_path=image_path,
            history=history,
            system_prompt=system_prompt,
        )
        normalized_messages = normalize_messages_for_family(
            messages=messages,
            family=self.model_family,
            image_path=image_path,
        )
        text = apply_family_chat_template(
            processor=self.processor,
            messages=normalized_messages,
            family=self.model_family,
            add_generation_prompt=True,
            enable_thinking=enable_thinking and family_supports_thinking(self.model_family),
        )
        inputs = self._prepare_inputs(text=text, image_path=image_path)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation.max_new_tokens,
                do_sample=generation.temperature > 0,
                temperature=max(generation.temperature, 1e-5),
                top_p=generation.top_p,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_stream(
        self,
        prompt: str,
        image_path: str | None = None,
        history: Iterable[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        generation: GenerationConfig | None = None,
        enable_thinking: bool = False,
    ):
        generation = generation or GenerationConfig()
        messages = self._build_messages(
            prompt=prompt,
            image_path=image_path,
            history=history,
            system_prompt=system_prompt,
        )
        normalized_messages = normalize_messages_for_family(
            messages=messages,
            family=self.model_family,
            image_path=image_path,
        )
        text = apply_family_chat_template(
            processor=self.processor,
            messages=normalized_messages,
            family=self.model_family,
            add_generation_prompt=True,
            enable_thinking=enable_thinking and family_supports_thinking(self.model_family),
        )
        inputs = self._prepare_inputs(text=text, image_path=image_path)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = {
            **inputs,
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.temperature > 0,
            "temperature": max(generation.temperature, 1e-5),
            "top_p": generation.top_p,
            "streamer": streamer,
        }

        generation_error: list[Exception] = []

        def run_generation() -> None:
            try:
                self.model.generate(**kwargs)
            except Exception as exc:
                generation_error.append(exc)

        thread = threading.Thread(target=run_generation)
        thread.start()
        for token_text in streamer:
            yield token_text
        thread.join()
        if generation_error:
            raise generation_error[0]
