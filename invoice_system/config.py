from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


LLMProvider = Literal["openai", "xai"]


class ConfigurationError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    provider: LLMProvider
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Load LLM configuration from environment variables."""

        load_dotenv()

        provider = os.getenv("LLM_PROVIDER", "xai").strip().lower()

        if provider not in {"openai", "xai"}:
            raise ConfigurationError(
                "LLM_PROVIDER must be either 'openai' or 'xai'; "
                f"received {provider!r}."
            )

        default_models = {
            "openai": "gpt-4.1-mini",
            "xai": "grok-build-0.1",
        }

        model = os.getenv(
            "LLM_MODEL",
            default_models[provider],
        ).strip()

        if not model:
            raise ConfigurationError("LLM_MODEL cannot be empty.")

        try:
            temperature = float(
                os.getenv("LLM_TEMPERATURE", "0")
            )
        except ValueError as exc:
            raise ConfigurationError(
                "LLM_TEMPERATURE must be a number."
            ) from exc

        try:
            max_tokens = int(
                os.getenv("LLM_MAX_TOKENS", "2048")
            )
        except ValueError as exc:
            raise ConfigurationError(
                "LLM_MAX_TOKENS must be an integer."
            ) from exc

        try:
            timeout_seconds = float(
                os.getenv("LLM_TIMEOUT_SECONDS", "60")
            )
        except ValueError as exc:
            raise ConfigurationError(
                "LLM_TIMEOUT_SECONDS must be a number."
            ) from exc

        try:
            max_retries = int(
                os.getenv("LLM_MAX_RETRIES", "2")
            )
        except ValueError as exc:
            raise ConfigurationError(
                "LLM_MAX_RETRIES must be an integer."
            ) from exc

        if max_tokens <= 0:
            raise ConfigurationError(
                "LLM_MAX_TOKENS must be greater than zero."
            )

        if timeout_seconds <= 0:
            raise ConfigurationError(
                "LLM_TIMEOUT_SECONDS must be greater than zero."
            )

        if max_retries < 0:
            raise ConfigurationError(
                "LLM_MAX_RETRIES cannot be negative."
            )

        return cls(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )