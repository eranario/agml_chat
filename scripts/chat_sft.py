#!/usr/bin/env python3
from __future__ import annotations

import argparse


def parse_target_modules(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [m.strip() for m in raw.split(",") if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a HuggingFace VLM on AgML-derived JSONL chat data")

    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id/path (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--train-jsonl", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--val-jsonl", type=str, default=None, help="Optional path to val.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/sft")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Optional compute dtype override",
    )
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention / SDPA selection")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train examples loaded from train-jsonl",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on eval examples loaded from val-jsonl",
    )

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-metrics-export", action="store_true", help="Disable training metric CSV/chart export")
    parser.add_argument(
        "--live-metrics",
        action="store_true",
        help="Continuously refresh metrics CSV/chart artifacts while training is running",
    )
    parser.add_argument(
        "--live-metrics-every-n-logs",
        type=int,
        default=1,
        help="Refresh live metrics artifacts every N log events when --live-metrics is enabled",
    )

    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated target module names for LoRA",
    )

    args = parser.parse_args()
    from agml_chat.training import TrainConfig, run_training

    config = TrainConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        use_flash_attention=not args.no_flash_attn,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        export_metrics=not args.no_metrics_export,
        live_metrics=args.live_metrics,
        live_metrics_every_n_logs=args.live_metrics_every_n_logs,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_target_modules(args.lora_target_modules),
    )
    run_training(config)


if __name__ == "__main__":
    main()
