from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_PROMPT_CONFIG = {
    "system_prompt": (
        "You are an agricultural vision assistant. Use only visual evidence from the image "
        "when possible and answer with concise, actionable language."
    ),
    "classification_instruction": (
        "Classify the diagnosis for this plant image into exactly one option from these choices: {labels}. "
        "Return with the correct answer."
    ),
    "research_mode_system_prompt": (
        "You are in research mode. Explain your observations step-by-step, mention uncertainty, "
        "then provide a final concise answer."
    ),
    "inference_instruction": (
        "Analyze this agricultural image and answer the user question. If classification is requested, "
        "choose one diagnosis from: {labels}."
    ),
}


@dataclass(frozen=True)
class PromptSet:
    system_prompt: str
    classification_instruction: str
    research_mode_system_prompt: str
    inference_instruction: str

    def render_classification_instruction(self, labels: Iterable[str]) -> str:
        label_list = ", ".join(labels)
        return self.classification_instruction.format(labels=label_list)

    def render_inference_instruction(self, labels: Iterable[str]) -> str:
        label_list = ", ".join(labels)
        return self.inference_instruction.format(labels=label_list)



def load_prompt_set(path: str | None = None) -> PromptSet:
    config = DEFAULT_PROMPT_CONFIG.copy()
    if path:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        unknown_keys = set(user_config).difference(config)
        if unknown_keys:
            raise ValueError(
                "Unknown prompt config keys: "
                + ", ".join(sorted(unknown_keys))
            )
        config.update(user_config)
    return PromptSet(**config)
