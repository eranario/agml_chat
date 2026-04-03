from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from agml_chat.chat_template_adapter import (
    ModelFamily,
    apply_family_chat_template,
    normalize_messages_for_family,
)


@dataclass(frozen=True)
class VLMRecord:
    image_path: str
    messages: list[dict[str, Any]]
    label: str
    dataset: str
    labels: list[str]


class VisionChatJsonlDataset(Dataset):
    """Loads JSONL records created by agml_chat.agml_data.export_agml_chat_dataset."""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.records: list[VLMRecord] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.records.append(
                    VLMRecord(
                        image_path=row["image_path"],
                        messages=row["messages"],
                        label=row["label"],
                        dataset=row.get("dataset", "unknown"),
                        labels=row.get("labels", []),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image_path": record.image_path,
            "messages": record.messages,
            "label": record.label,
            "dataset": record.dataset,
            "labels": record.labels,
        }



def prepare_messages_for_template(messages: list[dict[str, Any]], image_path: str) -> list[dict[str, Any]]:
    """Backward-compatible generic normalizer wrapper."""
    return normalize_messages_for_family(
        messages=messages,
        family=ModelFamily.GENERIC,
        image_path=image_path,
    )



def apply_chat_template(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """Backward-compatible generic template wrapper."""
    return apply_family_chat_template(
        processor=processor,
        messages=messages,
        family=ModelFamily.GENERIC,
        add_generation_prompt=add_generation_prompt,
    )



def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")
