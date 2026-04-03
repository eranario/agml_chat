from __future__ import annotations

from dataclasses import dataclass

from agml_chat.engine import ChatEngine, GenerationConfig


@dataclass
class ResearchResult:
    draft: str
    final: str



def run_research_mode(
    engine: ChatEngine,
    prompt: str,
    image_path: str | None,
    system_prompt: str,
    generation: GenerationConfig,
) -> ResearchResult:
    draft = engine.generate(
        prompt=prompt,
        image_path=image_path,
        system_prompt=system_prompt,
        generation=generation,
    )

    refinement_prompt = (
        "Refine the following draft answer into a concise final response. "
        "Preserve factual points, remove repetition, and keep uncertainty statements when relevant.\n\n"
        f"Original user request: {prompt}\n"
        f"Draft answer: {draft}"
    )

    final = engine.generate(
        prompt=refinement_prompt,
        image_path=image_path,
        system_prompt=system_prompt,
        generation=generation,
    )
    return ResearchResult(draft=draft, final=final)
