from typing import Any

import pytest

from invoice_system.config import ConfigurationError, LLMSettings
from invoice_system.llm import create_chat_model

# verifies that the xAI configuration reaches LangChain correctly without making a paid API request.
def make_xai_settings() -> LLMSettings:
    return LLMSettings(
        provider="xai",
        model="grok-build-0.1",
        temperature=0,
        max_tokens=2048,
        timeout_seconds=60,
        max_retries=2,
    )


def test_create_chat_model_initializes_xai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAI_API_KEY",
        "test-xai-api-key",
    )

    captured_arguments: dict[str, Any] = {}
    fake_model = object()

    def fake_init_chat_model(
        **kwargs: Any,
    ) -> object:
        captured_arguments.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        "invoice_system.llm.init_chat_model",
        fake_init_chat_model,
    )

    result = create_chat_model(make_xai_settings())

    assert result is fake_model
    assert captured_arguments == {
        "model": "grok-build-0.1",
        "model_provider": "xai",
        "temperature": 0,
        "max_tokens": 2048,
        "timeout": 60,
        "max_retries": 2,
    }


def test_create_chat_model_requires_xai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "XAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ConfigurationError,
        match="XAI_API_KEY is required",
    ):
        create_chat_model(make_xai_settings())