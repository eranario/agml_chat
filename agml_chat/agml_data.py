from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agml.data import AgMLDataLoader, public_data_sources

from agml_chat.common import ensure_dir
from agml_chat.prompts import PromptSet


@dataclass(frozen=True)
class AgMLDatasetInfo:
    name: str
    num_images: int
    classes: list[str]
    location: str


@dataclass(frozen=True)
class AgMLExample:
    dataset: str
    image_path: str
    label_id: int
    label_text: str
    raw_label_text: str
    crop_type: str
    class_name: str
    all_labels: list[str]
    all_diagnoses: list[str]


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.9
    val: float = 0.1
    test: float = 0.0

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


_AGML_CITATION_PATCHED = False


def _is_agml_citation_bug(exc: BaseException) -> bool:
    return "license_more_info" in str(exc)


def _patch_agml_citation_bug() -> bool:
    global _AGML_CITATION_PATCHED
    if _AGML_CITATION_PATCHED:
        return True

    try:
        from agml.utils import data as agml_utils_data
        from agml.utils import downloads as agml_utils_downloads
    except Exception:
        logging.exception("Failed to import AgML utils for citation bug patch.")
        return False

    original_copyright_print = agml_utils_data.copyright_print

    def _safe_copyright_print(name: str, location: str | None = None) -> None:
        try:
            original_copyright_print(name, location=location)
        except UnboundLocalError as exc:
            if not _is_agml_citation_bug(exc):
                raise
            logging.warning(
                "Suppressed AgML citation rendering bug for dataset '%s' (missing license_more_info).",
                name,
            )

    agml_utils_data.copyright_print = _safe_copyright_print
    agml_utils_downloads.copyright_print = _safe_copyright_print
    _AGML_CITATION_PATCHED = True
    return True



def list_classification_datasets(min_images: int = 0) -> list[AgMLDatasetInfo]:
    datasets = public_data_sources(ml_task="image_classification")
    out: list[AgMLDatasetInfo] = []
    for metadata in datasets:
        n_images = int(metadata.num_images)
        if n_images < min_images:
            continue
        country = metadata.location.country if metadata.location else "unknown"
        out.append(
            AgMLDatasetInfo(
                name=metadata.name,
                num_images=n_images,
                classes=list(metadata.classes),
                location=country,
            )
        )
    return sorted(out, key=lambda d: d.name)



def _resolve_local_dataset_root(dataset_name: str, dataset_path: str | None = None) -> Path:
    if dataset_path:
        base = Path(dataset_path).expanduser().resolve()
        if base.name == dataset_name:
            return base
        return base / dataset_name
    return Path.home() / ".agml" / "datasets" / dataset_name


def _create_agml_loader(dataset_name: str, dataset_path: str | None = None) -> AgMLDataLoader:
    loader_kwargs: dict[str, str] = {}
    if dataset_path:
        loader_kwargs["dataset_path"] = dataset_path

    try:
        return AgMLDataLoader(dataset_name, **loader_kwargs)
    except UnboundLocalError as exc:
        # Work around an upstream AgML bug triggered by some dataset citation metadata.
        if not _is_agml_citation_bug(exc):
            raise

        logging.warning(
            "AgML public loader hit citation bug for '%s'; patching citation print and retrying.",
            dataset_name,
        )
        if not _patch_agml_citation_bug():
            raise

        return AgMLDataLoader(dataset_name, **loader_kwargs)


def _load_dataset_examples(dataset_name: str, dataset_path: str | None = None, species_specific_options: bool = True) -> list[AgMLExample]:
    loader = _create_agml_loader(dataset_name, dataset_path=dataset_path)

    # AgML keeps a mapping of absolute image path -> class index for classification datasets.
    mapping = loader._builder.get_contents()  # noqa: SLF001
    class_lookup = dict(loader.num_to_class)
    sentence_lookup: dict[int, str] = {}
    parsed_lookup: dict[int, tuple[str, str]] = {}
    for idx in sorted(class_lookup):
        sentence, crop_type, class_name = _format_label_sentence(class_lookup[idx])
        sentence_lookup[idx] = sentence
        parsed_lookup[idx] = (crop_type, class_name)
    labels = [sentence_lookup[idx] for idx in sorted(sentence_lookup)]
    diagnoses = _dedupe_preserving_order([parsed_lookup[idx][1] for idx in sorted(parsed_lookup)])

    examples: list[AgMLExample] = []
    for image_path, label_idx in mapping.items():
        label_idx = int(label_idx)
        label_text = sentence_lookup[label_idx]
        raw_label_text = str(class_lookup[label_idx])
        crop_type, class_name = parsed_lookup[label_idx]
        
        if species_specific_options:
            example_diagnoses = _dedupe_preserving_order(
                [parsed_lookup[idx][1] for idx in sorted(parsed_lookup) if parsed_lookup[idx][0] == crop_type]
            )
        else:
            example_diagnoses = diagnoses

        examples.append(
            AgMLExample(
                dataset=dataset_name,
                image_path=str(image_path),
                label_id=label_idx,
                label_text=str(label_text),
                raw_label_text=raw_label_text,
                crop_type=crop_type,
                class_name=class_name,
                all_labels=labels,
                all_diagnoses=example_diagnoses,
            )
        )
    return examples



def _split_examples(
    examples: list[AgMLExample],
    split_ratios: SplitRatios,
    seed: int,
) -> dict[str, list[AgMLExample]]:
    rng = random.Random(seed)
    grouped: dict[int, list[AgMLExample]] = defaultdict(list)
    for example in examples:
        grouped[example.label_id].append(example)

    split_out: dict[str, list[AgMLExample]] = {"train": [], "val": [], "test": []}
    for _, group in grouped.items():
        rng.shuffle(group)
        n = len(group)
        counts = {
            "train": int(n * split_ratios.train),
            "val": int(n * split_ratios.val),
            "test": int(n * split_ratios.test),
        }
        assigned = counts["train"] + counts["val"] + counts["test"]
        remainder = n - assigned
        if remainder > 0:
            priority = sorted(
                (
                    ("train", split_ratios.train),
                    ("val", split_ratios.val),
                    ("test", split_ratios.test),
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for i in range(remainder):
                split_name = priority[i % len(priority)][0]
                counts[split_name] += 1

        n_train = counts["train"]
        n_val = counts["val"]
        n_test = counts["test"]

        train_slice = group[:n_train]
        val_slice = group[n_train : n_train + n_val]
        test_slice = group[n_train + n_val : n_train + n_val + n_test]

        split_out["train"].extend(train_slice)
        split_out["val"].extend(val_slice)
        split_out["test"].extend(test_slice)

    for split in split_out:
        rng.shuffle(split_out[split])
    return split_out



def build_agml_splits(
    dataset_names: Iterable[str],
    split_ratios: SplitRatios,
    seed: int = 42,
    max_samples_per_dataset: int | None = None,
    dataset_path: str | None = None,
    species_specific_options: bool = True,
) -> dict[str, list[AgMLExample]]:
    merged = {"train": [], "val": [], "test": []}
    for dataset_name in dataset_names:
        dataset_examples = _load_dataset_examples(
            dataset_name, 
            dataset_path=dataset_path, 
            species_specific_options=species_specific_options
        )
        if max_samples_per_dataset and len(dataset_examples) > max_samples_per_dataset:
            rng = random.Random(seed)
            dataset_examples = rng.sample(dataset_examples, max_samples_per_dataset)
        split = _split_examples(dataset_examples, split_ratios=split_ratios, seed=seed)
        for key in merged:
            merged[key].extend(split[key])
    return merged



def _record_from_example(example: AgMLExample, prompt_set: PromptSet) -> dict:
    instruction = _build_species_diagnosis_instruction(example=example, prompt_set=prompt_set)
    return {
        "dataset": example.dataset,
        "image_path": example.image_path,
        "label": example.label_text,
        "raw_label": example.raw_label_text,
        "crop_type": example.crop_type,
        "class_name": example.class_name,
        "labels": _dedupe_preserving_order(example.all_diagnoses),
        "messages": [
            {"role": "system", "content": prompt_set.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": example.label_text},
        ],
    }


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_species_diagnosis_instruction(example: AgMLExample, prompt_set: PromptSet) -> str:
    species_context = f'This is an image of a "{example.crop_type}" plant.'
    diagnosis_instruction = prompt_set.render_classification_instruction(
        _dedupe_preserving_order(example.all_diagnoses)
    )
    return (
        f"{species_context}\n"
        f"{diagnosis_instruction}\n"
        'Respond exactly in this format: This is an image of a "<crop type>" with "<class>".'
    )


def _clean_label_component(value: str) -> str:
    value = value.replace("_", " ")
    value = value.replace(",", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _split_crop_and_class(raw_label: str) -> tuple[str, str]:
    parts = re.split(r"_{2,}", raw_label, maxsplit=1)
    if len(parts) == 2:
        crop = _clean_label_component(parts[0])
        class_name = _clean_label_component(parts[1])
        return crop, class_name

    cleaned = _clean_label_component(raw_label)
    if "background" in cleaned.lower():
        return cleaned, "background"
    return cleaned, "unspecified class"


def _format_label_sentence(raw_label: str) -> tuple[str, str, str]:
    crop_type, class_name = _split_crop_and_class(raw_label)
    sentence = f'This is an image of a "{crop_type}" with "{class_name}".'
    return sentence, crop_type, class_name



def write_jsonl(records: Iterable[dict], output_path: str) -> int:
    path = Path(output_path)
    ensure_dir(str(path.parent))
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count



def export_agml_chat_dataset(
    dataset_names: Iterable[str],
    output_dir: str,
    prompt_set: PromptSet,
    split_ratios: SplitRatios,
    seed: int = 42,
    max_samples_per_dataset: int | None = None,
    dataset_path: str | None = None,
    species_specific_options: bool = True,
) -> dict[str, str]:
    splits = build_agml_splits(
        dataset_names=dataset_names,
        split_ratios=split_ratios,
        seed=seed,
        max_samples_per_dataset=max_samples_per_dataset,
        dataset_path=dataset_path,
        species_specific_options=species_specific_options,
    )

    ensure_dir(output_dir)
    output_paths: dict[str, str] = {}
    for split_name, examples in splits.items():
        if not examples:
            continue
        records = [_record_from_example(example, prompt_set=prompt_set) for example in examples]
        out_path = str(Path(output_dir) / f"{split_name}.jsonl")
        write_jsonl(records, out_path)
        output_paths[split_name] = out_path

    metadata_path = Path(output_dir) / "dataset_manifest.json"
    manifest = {
        "datasets": list(dataset_names),
        "split_ratios": {
            "train": split_ratios.train,
            "val": split_ratios.val,
            "test": split_ratios.test,
        },
        "seed": seed,
        "max_samples_per_dataset": max_samples_per_dataset,
        "files": output_paths,
    }
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_paths["manifest"] = str(metadata_path)
    return output_paths
