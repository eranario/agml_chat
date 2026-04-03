from __future__ import annotations

import logging
from inspect import signature
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments
from transformers import __version__ as transformers_version

from agml_chat.chat_template_adapter import (
    ModelFamily,
    apply_family_chat_template,
    normalize_messages_for_family,
)
from agml_chat.common import build_runtime_config, configure_logging, ensure_dir, set_seed
from agml_chat.dataset import VisionChatJsonlDataset, load_image
from agml_chat.modeling import load_model_and_processor, maybe_wrap_lora


@dataclass
class TrainConfig:
    model_name: str
    train_jsonl: str
    output_dir: str
    val_jsonl: str | None = None
    seed: int = 42
    device: str = "auto"
    dtype: str | None = None
    use_flash_attention: bool = True
    trust_remote_code: bool = True

    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 2
    max_length: int = 2048

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3

    gradient_checkpointing: bool = True

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


class VisionLanguageSFTCollator:
    def __init__(self, processor: Any, model_family: ModelFamily, max_length: int = 2048):
        self.processor = processor
        self.model_family = model_family
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Any] = []

        for feature in features:
            messages = normalize_messages_for_family(
                messages=feature["messages"],
                family=self.model_family,
                image_path=feature["image_path"],
            )
            text = apply_family_chat_template(
                processor=self.processor,
                messages=messages,
                family=self.model_family,
                add_generation_prompt=False,
            )
            image = load_image(feature["image_path"])
            texts.append(text)
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        pad_token_id = None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        labels[labels == pad_token_id] = -100

        batch["labels"] = labels
        return batch


def _build_training_arguments(**kwargs: Any) -> TrainingArguments:
    """
    Build TrainingArguments across transformers versions by:
    - mapping renamed params (`evaluation_strategy` <-> `eval_strategy`)
    - dropping unsupported params with a warning
    """
    params = signature(TrainingArguments.__init__).parameters
    resolved: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in params:
            resolved[key] = value
            continue

        if key == "evaluation_strategy" and "eval_strategy" in params:
            resolved["eval_strategy"] = value
            continue
        if key == "eval_strategy" and "evaluation_strategy" in params:
            resolved["evaluation_strategy"] = value
            continue

        logging.warning(
            "Dropping unsupported TrainingArguments param '%s' for transformers %s.",
            key,
            transformers_version,
        )

    return TrainingArguments(**resolved)



def run_training(config: TrainConfig) -> None:
    configure_logging()
    set_seed(config.seed)

    runtime = build_runtime_config(device=config.device, dtype=config.dtype)

    model, processor, model_family = load_model_and_processor(
        model_name=config.model_name,
        runtime=runtime,
        use_flash_attention=config.use_flash_attention,
        trust_remote_code=config.trust_remote_code,
    )

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = maybe_wrap_lora(
        model,
        enabled=config.use_lora,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )

    train_dataset = VisionChatJsonlDataset(config.train_jsonl)
    eval_dataset = None
    if config.val_jsonl:
        val_path = Path(config.val_jsonl)
        if val_path.exists():
            eval_dataset = VisionChatJsonlDataset(config.val_jsonl)
        else:
            logging.warning(
                "Validation JSONL not found at '%s'; continuing with train-only (evaluation disabled).",
                config.val_jsonl,
            )

    collator = VisionLanguageSFTCollator(
        processor=processor,
        model_family=model_family,
        max_length=config.max_length,
    )

    ensure_dir(config.output_dir)
    report_to = []
    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = _build_training_arguments(
        output_dir=config.output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        bf16=runtime.device == "cuda" and runtime.torch_dtype == torch.bfloat16,
        fp16=runtime.device == "cuda" and runtime.torch_dtype == torch.float16,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    final_dir = str(Path(config.output_dir) / "final")
    ensure_dir(final_dir)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
