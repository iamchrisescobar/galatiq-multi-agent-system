from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from invoice_system.config import ConfigurationError, LLMSettings


PROVIDER_API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
}


def create_chat_model(settings: LLMSettings) -> BaseChatModel:
    """
    Create the chat model used by the invoice agents.

    Agent code depends only on LangChain's BaseChatModel interface. Changing
    provider therefore requires configuration rather than changes to the
    agents themselves.
    """

    api_key_variable = PROVIDER_API_KEY_VARIABLES[settings.provider]

    if not os.getenv(api_key_variable):
        raise ConfigurationError(
            f"{api_key_variable} is required when "
            f"LLM_PROVIDER={settings.provider!r}."
        )

    return init_chat_model(
        model=settings.model,
        model_provider=settings.provider,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )