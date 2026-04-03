from __future__ import annotations

from agml_chat.agml_data import _format_label_sentence, _split_crop_and_class


def test_split_crop_and_class_plant_village_style() -> None:
    crop, cls = _split_crop_and_class("Grape___Leaf_blight")
    assert crop == "Grape"
    assert cls == "Leaf blight"


def test_format_label_sentence() -> None:
    sentence, crop, cls = _format_label_sentence("Tomato___Early_blight")
    assert crop == "Tomato"
    assert cls == "Early blight"
    assert sentence == 'This is an image of a "Tomato" with "Early blight".'
