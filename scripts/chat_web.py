#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agml-chat FastAPI web server")
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
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    from agml_chat.engine import ChatEngine
    from agml_chat.prompts import load_prompt_set
    from agml_chat.web import create_app

    prompt_set = load_prompt_set(args.prompt_config)
    engine = ChatEngine.from_pretrained(
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
        use_flash_attention=not args.no_flash_attn,
    )

    ui_path = str(Path(__file__).resolve().parent.parent / "agml_chat" / "templates" / "ui.html")
    app = create_app(engine=engine, prompt_set=prompt_set, ui_html_path=ui_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
