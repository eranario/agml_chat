from __future__ import annotations

import pytest

from agml_chat import agml_data


def test_create_agml_loader_patches_and_retries_on_license_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_name = "plant_village_classification"
    init_calls: list[tuple[tuple, dict]] = []
    patch_calls = {"count": 0}

    class BrokenPublicLoader:
        def __init__(self, *args, **kwargs):
            init_calls.append((args, kwargs))
            if len(init_calls) == 1:
                raise UnboundLocalError("cannot access local variable 'license_more_info'")
            self.name = args[0]
            self.dataset_path = kwargs.get("dataset_path")

    monkeypatch.setattr(agml_data, "AgMLDataLoader", BrokenPublicLoader)
    monkeypatch.setattr(agml_data, "_AGML_CITATION_PATCHED", False)

    def _fake_patch() -> bool:
        patch_calls["count"] += 1
        return True

    monkeypatch.setattr(agml_data, "_patch_agml_citation_bug", _fake_patch)

    loader = agml_data._create_agml_loader(dataset_name, dataset_path="/tmp/agml")
    assert loader.name == dataset_name
    assert loader.dataset_path == "/tmp/agml"
    assert len(init_calls) == 2
    assert patch_calls["count"] == 1


def test_create_agml_loader_propagates_other_unboundlocalerror(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPublicLoader:
        def __init__(self, *args, **kwargs):
            raise UnboundLocalError("some_other_issue")

    monkeypatch.setattr(agml_data, "AgMLDataLoader", BrokenPublicLoader)

    with pytest.raises(UnboundLocalError):
        agml_data._create_agml_loader("plant_village_classification")


def test_create_agml_loader_raises_if_patch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPublicLoader:
        def __init__(self, *args, **kwargs):
            raise UnboundLocalError("cannot access local variable 'license_more_info'")

    monkeypatch.setattr(agml_data, "AgMLDataLoader", BrokenPublicLoader)
    monkeypatch.setattr(agml_data, "_AGML_CITATION_PATCHED", False)
    monkeypatch.setattr(agml_data, "_patch_agml_citation_bug", lambda: False)

    with pytest.raises(UnboundLocalError):
        agml_data._create_agml_loader("plant_village_classification")
