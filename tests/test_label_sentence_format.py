from __future__ import annotations

from agml_chat.agml_data import AgMLExample, _build_species_diagnosis_instruction, _format_label_sentence, _split_crop_and_class
from agml_chat.prompts import load_prompt_set


def test_split_crop_and_class_plant_village_style() -> None:
    crop, cls = _split_crop_and_class("Grape___Leaf_blight")
    assert crop == "Grape"
    assert cls == "Leaf blight"


def test_format_label_sentence() -> None:
    sentence, crop, cls = _format_label_sentence("Tomato___Early_blight")
    assert crop == "Tomato"
    assert cls == "Early blight"
    assert sentence == 'This is an image of a "Tomato" with "Early blight".'


def test_species_first_instruction_uses_diagnosis_choices_only() -> None:
    prompt_set = load_prompt_set()
    example = AgMLExample(
        dataset="plant_village_classification",
        image_path="/tmp/image.jpg",
        label_id=0,
        label_text='This is an image of a "Tomato" with "Early blight".',
        raw_label_text="Tomato___Early_blight",
        crop_type="Tomato",
        class_name="Early blight",
        all_labels=[
            'This is an image of a "Tomato" with "Early blight".',
            'This is an image of a "Tomato" with "Late blight".',
        ],
        all_diagnoses=["Early blight", "Late blight"],
    )

    instruction = _build_species_diagnosis_instruction(example=example, prompt_set=prompt_set)
    lines = instruction.splitlines()

    assert lines[0] == 'This is an image of a "Tomato" plant.'
    assert "Early blight" in lines[1]
    assert "Late blight" in lines[1]
    assert "This is an image of a" not in lines[1]
    assert 'Respond exactly in this format: This is an image of a "<crop type>" with "<class>".' in lines[2]
