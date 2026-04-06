#!/usr/bin/env python3
from __future__ import annotations

import argparse


HELP_TEXT = """
Commands:
  /quit                 Exit
  /clear                Clear conversation history
  /research on|off      Toggle research mode
  /image <path>         Set default image path for subsequent prompts
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal CLI chat for fine-tuned AgML VLMs")
    parser.add_argument("--model", type=str, required=True, help="Model id/path")
    parser.add_argument("--prompt-config", type=str, default=None, help="Optional prompt YAML")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Optional compute dtype override",
    )
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Gemma 4 thinking mode when supported")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image", type=str, default=None, help="Optional image path for one-shot mode")
    parser.add_argument("--research", action="store_true", help="Enable research mode")
    parser.add_argument("--single-prompt", type=str, default=None, help="Run one prompt and exit")
    args = parser.parse_args()
    from agml_chat.engine import ChatEngine, GenerationConfig
    from agml_chat.prompts import load_prompt_set
    from agml_chat.research import run_research_mode

    prompt_set = load_prompt_set(args.prompt_config)
    engine = ChatEngine.from_pretrained(
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
        use_flash_attention=not args.no_flash_attn,
    )

    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    history: list[dict[str, str]] = []
    default_image = args.image
    research_mode = args.research

    def answer(prompt: str, image_path: str | None) -> str:
        if research_mode:
            result = run_research_mode(
                engine=engine,
                prompt=prompt,
                image_path=image_path,
                system_prompt=prompt_set.research_mode_system_prompt,
                generation=generation,
            )
            return result.final
        return engine.generate(
            prompt=prompt,
            image_path=image_path,
            history=history,
            system_prompt=prompt_set.system_prompt,
            generation=generation,
            enable_thinking=args.enable_thinking,
        )

    if args.single_prompt:
        print(answer(args.single_prompt, default_image))
        return

    print("agml-chat CLI")
    print(HELP_TEXT)
    while True:
        try:
            raw = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not raw:
            continue

        if raw == "/quit":
            print("bye")
            break

        if raw == "/clear":
            history.clear()
            print("history cleared")
            continue

        if raw.startswith("/research"):
            _, _, flag = raw.partition(" ")
            research_mode = flag.strip().lower() == "on"
            print(f"research mode: {'on' if research_mode else 'off'}")
            continue

        if raw.startswith("/image"):
            _, _, image_path = raw.partition(" ")
            default_image = image_path.strip() or None
            print(f"default image: {default_image}")
            continue

        response = answer(raw, default_image)
        print(f"assistant> {response}")
        history.append({"role": "user", "content": raw})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
