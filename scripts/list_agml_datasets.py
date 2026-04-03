#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="List AgML image-classification datasets")
    parser.add_argument("--min-images", type=int, default=0, help="Minimum number of images")
    args = parser.parse_args()
    from agml_chat.agml_data import list_classification_datasets

    datasets = list_classification_datasets(min_images=args.min_images)
    if not datasets:
        print("No classification datasets found.")
        return

    print(f"Found {len(datasets)} classification datasets:\n")
    for ds in datasets:
        print(f"- {ds.name} | images={ds.num_images} | classes={len(ds.classes)} | location={ds.location}")


if __name__ == "__main__":
    main()
