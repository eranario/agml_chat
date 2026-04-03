#!/usr/bin/env python3
from __future__ import annotations

import argparse


def parse_datasets(raw: str) -> list[str]:
    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AgML image-classification data for VLM SFT")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated AgML dataset names (e.g. plant_village_classification,rice_leaf_disease_classification)",
    )
    parser.add_argument("--output-dir", type=str, default="data/agml_sft", help="Output directory for JSONL files")
    parser.add_argument("--prompt-config", type=str, default=None, help="Optional YAML file for prompt templates")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional local AgML datasets root path")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional cap per dataset before splitting",
    )
    args = parser.parse_args()
    from agml_chat.agml_data import SplitRatios, export_agml_chat_dataset
    from agml_chat.prompts import load_prompt_set

    datasets = parse_datasets(args.datasets)
    split_ratios = SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    prompt_set = load_prompt_set(args.prompt_config)

    paths = export_agml_chat_dataset(
        dataset_names=datasets,
        output_dir=args.output_dir,
        prompt_set=prompt_set,
        split_ratios=split_ratios,
        seed=args.seed,
        max_samples_per_dataset=args.max_samples_per_dataset,
        dataset_path=args.dataset_path,
    )

    print("Prepared dataset files:")
    for split_name, file_path in paths.items():
        print(f"- {split_name}: {file_path}")


if __name__ == "__main__":
    main()
