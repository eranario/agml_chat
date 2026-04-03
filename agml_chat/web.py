from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agml_chat.engine import ChatEngine, GenerationConfig
from agml_chat.prompts import PromptSet
from agml_chat.research import run_research_mode

# Abuse-prevention defaults inspired by nanochat web serving constraints.
MAX_MESSAGES_PER_REQUEST = 200
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MAX_NEW_TOKENS = 1
MAX_MAX_NEW_TOKENS = 4096


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    image_data_url: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 256
    research_mode: bool = False


def _validate_chat_request(request: ChatCompletionRequest) -> None:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} are allowed.",
        )

    total_chars = 0
    for idx, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {idx} is empty.")
        msg_len = len(message.content)
        if msg_len > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {idx} exceeds max length ({MAX_MESSAGE_LENGTH}).",
            )
        total_chars += msg_len

    if total_chars > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation exceeds max total length ({MAX_TOTAL_CONVERSATION_LENGTH}).",
        )

    if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(
            status_code=400,
            detail=f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}.",
        )
    if not (MIN_TOP_P <= request.top_p <= MAX_TOP_P):
        raise HTTPException(
            status_code=400,
            detail=f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}.",
        )
    if not (MIN_MAX_NEW_TOKENS <= request.max_new_tokens <= MAX_MAX_NEW_TOKENS):
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens must be between {MIN_MAX_NEW_TOKENS} and {MAX_MAX_NEW_TOKENS}.",
        )


def _extract_prompt_and_history(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    if not messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    if messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must be from 'user'")

    prompt = messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
    return prompt, history



def _decode_image_data_url(image_data_url: str | None) -> str | None:
    if not image_data_url:
        return None

    if "," not in image_data_url:
        raise HTTPException(status_code=400, detail="Invalid image data URL")

    header, payload = image_data_url.split(",", 1)
    if "base64" not in header:
        raise HTTPException(status_code=400, detail="Only base64 image data URLs are supported")

    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc
    suffix = ".png"
    if "image/jpeg" in header:
        suffix = ".jpg"
    if "image/webp" in header:
        suffix = ".webp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name



def create_app(engine: ChatEngine, prompt_set: PromptSet, ui_html_path: str) -> FastAPI:
    app = FastAPI(title="agml-chat")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return Path(ui_html_path).read_text(encoding="utf-8")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict:
        _validate_chat_request(request)
        prompt, history = _extract_prompt_and_history(request.messages)
        image_path = _decode_image_data_url(request.image_data_url)

        generation = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        if request.research_mode:
            result = run_research_mode(
                engine=engine,
                prompt=prompt,
                image_path=image_path,
                system_prompt=prompt_set.research_mode_system_prompt,
                generation=generation,
            )
            answer = result.final
            draft = result.draft
        else:
            answer = engine.generate(
                prompt=prompt,
                image_path=image_path,
                history=history,
                system_prompt=prompt_set.system_prompt,
                generation=generation,
            )
            draft = None

        return {
            "message": {"role": "assistant", "content": answer},
            "research_draft": draft,
        }

    @app.post("/chat/completions/stream")
    async def chat_stream(request: ChatCompletionRequest) -> StreamingResponse:
        _validate_chat_request(request)
        prompt, history = _extract_prompt_and_history(request.messages)
        image_path = _decode_image_data_url(request.image_data_url)

        generation = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        async def token_stream() -> AsyncGenerator[str, None]:
            if request.research_mode:
                result = run_research_mode(
                    engine=engine,
                    prompt=prompt,
                    image_path=image_path,
                    system_prompt=prompt_set.research_mode_system_prompt,
                    generation=generation,
                )
                for token in result.final.split(" "):
                    yield f"data: {json.dumps({'token': token + ' '})}\n\n"
                yield "data: {\"done\": true}\n\n"
                return

            for token in engine.generate_stream(
                prompt=prompt,
                image_path=image_path,
                history=history,
                system_prompt=prompt_set.system_prompt,
                generation=generation,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: {\"done\": true}\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    return app
