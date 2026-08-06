import pytest

from invoice_system.config import ConfigurationError, LLMSettings


def test_llm_settings_load_openai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-openai-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")

    settings = LLMSettings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "test-openai-model"
    assert settings.temperature == 0
    assert settings.timeout_seconds == 30
    assert settings.max_retries == 1


def test_llm_settings_load_xai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("LLM_MODEL", "test-grok-model")

    settings = LLMSettings.from_env()

    assert settings.provider == "xai"
    assert settings.model == "test-grok-model"


def test_llm_settings_reject_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")

    with pytest.raises(ConfigurationError, match="openai.*xai"):
        LLMSettings.from_env()


def test_llm_settings_reject_invalid_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-an-integer")

    with pytest.raises(ConfigurationError, match="must be an integer"):
        LLMSettings.from_env()