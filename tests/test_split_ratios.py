from __future__ import annotations

from agml_chat.agml_data import AgMLExample, SplitRatios, _split_examples


def _make_examples(n: int, label_id: int = 0) -> list[AgMLExample]:
    return [
        AgMLExample(
            dataset="d",
            image_path=f"/tmp/{i}.jpg",
            label_id=label_id,
            label_text="class_a",
            all_labels=["class_a"],
        )
        for i in range(n)
    ]


def test_split_respects_zero_test_ratio() -> None:
    examples = _make_examples(20)
    splits = _split_examples(examples, split_ratios=SplitRatios(train=0.8, val=0.2, test=0.0), seed=42)
    assert len(splits["test"]) == 0
    assert len(splits["train"]) + len(splits["val"]) == 20
