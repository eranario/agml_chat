from __future__ import annotations

import pytest
from fastapi import HTTPException

from agml_chat.web import ChatCompletionRequest, ChatMessage, _validate_chat_request


def test_validate_chat_request_rejects_empty_messages() -> None:
    request = ChatCompletionRequest(messages=[])
    with pytest.raises(HTTPException):
        _validate_chat_request(request)


def test_validate_chat_request_rejects_long_message() -> None:
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="x" * 9000)])
    with pytest.raises(HTTPException):
        _validate_chat_request(request)


def test_validate_chat_request_accepts_valid_payload() -> None:
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
    _validate_chat_request(request)
