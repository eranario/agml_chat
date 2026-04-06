#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "No Hugging Face token provided. Pass --token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a trained agml-chat model folder to Hugging Face Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local model directory to upload (e.g. runs/sft_20260403_123456/final)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub repo id in the format 'username/repo-name' or 'org/repo-name'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (optional if HF_TOKEN or HUGGINGFACE_HUB_TOKEN is set)",
    )
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not exist")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload agml-chat trained model",
        help="Commit message for Hub upload",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch/revision on the Hub repo",
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        default=None,
        help="Optional comma-separated include patterns for upload",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        default=".DS_Store,*.tmp,*.log,*.pt,optimizer.pt,scheduler.pt,rng_state.pth,trainer_state.json",
        help="Optional comma-separated ignore patterns",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist or is not a directory: {model_dir}")

    token = _resolve_token(args.token)
    api = HfApi(token=token)

    allow_patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()] if args.allow_patterns else None
    ignore_patterns = [p.strip() for p in args.ignore_patterns.split(",") if p.strip()] if args.ignore_patterns else None

    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        token=token,
        repo_type="model",
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    print("Upload complete.")
    print(f"Repo: https://huggingface.co/{args.repo_id}")
    print(f"Revision: {args.revision}")
    print(f"Commit: {commit_info.oid}")


if __name__ == "__main__":
    main()
