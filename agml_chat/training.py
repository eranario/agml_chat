from __future__ import annotations

import csv
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
    family_supports_thinking,
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
    export_metrics: bool = True


class VisionLanguageSFTCollator:
    def __init__(self, processor: Any, model_family: ModelFamily, max_length: int = 2048, enable_thinking: bool = False):
        self.processor = processor
        self.model_family = model_family
        self.max_length = max_length
        self.enable_thinking = enable_thinking

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
                enable_thinking=self.enable_thinking and family_supports_thinking(self.model_family),
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


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _collect_metric_series(log_history: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    excluded_keys = {
        "epoch",
        "step",
        "total_flos",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_tokens_per_second",
    }
    metric_series: dict[str, list[tuple[float, float]]] = {}

    for row in log_history:
        step = row.get("step")
        if not _is_scalar_number(step):
            continue

        for key, value in row.items():
            if key in excluded_keys:
                continue
            if not _is_scalar_number(value):
                continue
            metric_series.setdefault(key, []).append((float(step), float(value)))

    return metric_series


def _write_long_metrics_csv(path: Path, log_history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "epoch", "metric", "value"])
        for row in log_history:
            step = row.get("step")
            if not _is_scalar_number(step):
                continue
            epoch = row.get("epoch") if _is_scalar_number(row.get("epoch")) else ""
            for key, value in row.items():
                if key in {"step", "epoch"}:
                    continue
                if not _is_scalar_number(value):
                    continue
                writer.writerow([float(step), epoch, key, float(value)])


def _write_wide_metrics_csv(path: Path, metric_series: dict[str, list[tuple[float, float]]]) -> None:
    by_step: dict[float, dict[str, float]] = {}
    for metric, points in metric_series.items():
        for step, value in points:
            by_step.setdefault(step, {})[metric] = value

    metrics = sorted(metric_series)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *metrics])
        for step in sorted(by_step):
            row = [step]
            values = by_step[step]
            for metric in metrics:
                row.append(values.get(metric, ""))
            writer.writerow(row)


def _plot_metrics_dashboards(
    metrics_dir: Path,
    metric_series: dict[str, list[tuple[float, float]]],
    metrics_per_figure: int = 9,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping training metric plots.")
        return []

    preferred_order = [
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "train_loss",
        "eval_accuracy",
        "eval_f1",
        "eval_precision",
        "eval_recall",
    ]
    metric_names = sorted(metric_series.keys(), key=lambda name: (name not in preferred_order, name))

    output_paths: list[Path] = []
    for page, start in enumerate(range(0, len(metric_names), metrics_per_figure), start=1):
        names = metric_names[start : start + metrics_per_figure]
        cols = 3
        rows = (len(names) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows), squeeze=False)
        fig.suptitle("Training Metrics Dashboard", fontsize=14)

        for idx, metric in enumerate(names):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            points = sorted(metric_series[metric], key=lambda item: item[0])
            steps = [x for x, _ in points]
            values = [y for _, y in points]

            ax.plot(steps, values, linewidth=1.8, label=metric)
            if metric in {"loss", "eval_loss"} and len(values) >= 5:
                # Smoothed line makes trend interpretation easier on noisy curves.
                window = min(25, max(5, len(values) // 20))
                smooth = []
                for i in range(len(values)):
                    left = max(0, i - window + 1)
                    smooth.append(sum(values[left : i + 1]) / (i - left + 1))
                ax.plot(steps, smooth, linestyle="--", linewidth=1.2, label=f"{metric} (smooth)")

            ax.set_title(metric)
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        for idx in range(len(names), rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        fig.tight_layout()
        output = metrics_dir / f"metrics_dashboard_{page}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        output_paths.append(output)

    return output_paths


def _write_metrics_summary(path: Path, metric_series: dict[str, list[tuple[float, float]]]) -> None:
    lines = ["# Training Metrics Summary", ""]
    lines.append(f"- Metrics tracked: {len(metric_series)}")

    if "loss" in metric_series and metric_series["loss"]:
        min_loss_step, min_loss = min(metric_series["loss"], key=lambda item: item[1])
        last_step, last_loss = metric_series["loss"][-1]
        lines.append(f"- Best loss: {min_loss:.6f} at step {int(min_loss_step)}")
        lines.append(f"- Final logged loss: {last_loss:.6f} at step {int(last_step)}")

    if "eval_loss" in metric_series and metric_series["eval_loss"]:
        min_eval_step, min_eval = min(metric_series["eval_loss"], key=lambda item: item[1])
        lines.append(f"- Best eval_loss: {min_eval:.6f} at step {int(min_eval_step)}")

    if "learning_rate" in metric_series and metric_series["learning_rate"]:
        last_lr = metric_series["learning_rate"][-1][1]
        lines.append(f"- Final learning_rate: {last_lr:.8f}")

    lines.extend(["", "## Available Files", "", "- `metrics_long.csv`: long-form table with one metric per row", "- `metrics_wide.csv`: step-indexed table suitable for spreadsheets", "- `metrics_dashboard_*.png`: dashboard charts grouped across metrics"]) 

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_training_metrics(log_history: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metric_series = _collect_metric_series(log_history)
    long_csv = metrics_dir / "metrics_long.csv"
    wide_csv = metrics_dir / "metrics_wide.csv"
    summary_md = metrics_dir / "training_summary.md"

    _write_long_metrics_csv(long_csv, log_history)
    _write_wide_metrics_csv(wide_csv, metric_series)
    dashboards = _plot_metrics_dashboards(metrics_dir, metric_series)
    _write_metrics_summary(summary_md, metric_series)

    return {
        "metrics_dir": metrics_dir,
        "long_csv": long_csv,
        "wide_csv": wide_csv,
        "dashboards": dashboards,
        "summary": summary_md,
        "metric_count": len(metric_series),
    }


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

    if config.export_metrics:
        metrics_output = export_training_metrics(
            log_history=trainer.state.log_history,
            output_dir=Path(config.output_dir) / "metrics",
        )
        logging.info(
            "Exported %d metrics to %s",
            metrics_output["metric_count"],
            metrics_output["metrics_dir"],
        )

    final_dir = str(Path(config.output_dir) / "final")
    ensure_dir(final_dir)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
