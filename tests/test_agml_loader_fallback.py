from __future__ import annotations

from pathlib import Path

import pytest

from agml_chat import agml_data


def test_create_agml_loader_falls_back_to_custom_on_license_bug(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_name = "plant_village_classification"
    dataset_dir = tmp_path / dataset_name
    dataset_dir.mkdir(parents=True)

    class BrokenPublicLoader:
        def __init__(self, *args, **kwargs):
            raise UnboundLocalError("cannot access local variable 'license_more_info'")

        @staticmethod
        def custom(name, dataset_path=None):
            return {"name": name, "dataset_path": dataset_path}

    monkeypatch.setattr(agml_data, "AgMLDataLoader", BrokenPublicLoader)

    loader = agml_data._create_agml_loader(dataset_name, dataset_path=str(tmp_path))
    assert loader["name"] == dataset_name
    assert loader["dataset_path"] == str(dataset_dir)


def test_create_agml_loader_propagates_other_unboundlocalerror(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPublicLoader:
        def __init__(self, *args, **kwargs):
            raise UnboundLocalError("some_other_issue")

        @staticmethod
        def custom(name, dataset_path=None):
            return {"name": name, "dataset_path": dataset_path}

    monkeypatch.setattr(agml_data, "AgMLDataLoader", BrokenPublicLoader)

    with pytest.raises(UnboundLocalError):
        agml_data._create_agml_loader("plant_village_classification")
