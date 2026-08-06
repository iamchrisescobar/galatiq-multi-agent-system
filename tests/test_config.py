import pytest

from invoice_system.config import ConfigurationError, LLMSettings


LLM_ENVIRONMENT_VARIABLES = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
)


def clear_llm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Clear model settings and prevent the developer's local .env file from
    affecting default-value tests.
    """

    monkeypatch.setattr(
        "invoice_system.config.load_dotenv",
        lambda: None,
    )

    for variable in LLM_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_llm_settings_default_to_low_cost_xai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_llm_environment(monkeypatch)

    settings = LLMSettings.from_env()

    assert settings.provider == "xai"
    assert settings.model == "grok-build-0.1"
    assert settings.temperature == 0
    assert settings.max_tokens == 2048
    assert settings.timeout_seconds == 60
    assert settings.max_retries == 2


def test_llm_settings_load_openai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-openai-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1000")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")

    settings = LLMSettings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "test-openai-model"
    assert settings.temperature == 0
    assert settings.max_tokens == 1000
    assert settings.timeout_seconds == 30
    assert settings.max_retries == 1


def test_llm_settings_load_xai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv(
        "LLM_MODEL",
        "test-grok-model",
    )

    settings = LLMSettings.from_env()

    assert settings.provider == "xai"
    assert settings.model == "test-grok-model"


def test_llm_settings_reject_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "unknown-provider",
    )

    with pytest.raises(
        ConfigurationError,
        match="openai.*xai",
    ):
        LLMSettings.from_env()


def test_llm_settings_reject_invalid_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_MAX_RETRIES",
        "not-an-integer",
    )

    with pytest.raises(
        ConfigurationError,
        match="must be an integer",
    ):
        LLMSettings.from_env()


def test_llm_settings_reject_invalid_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_MAX_TOKENS",
        "not-an-integer",
    )

    with pytest.raises(
        ConfigurationError,
        match="LLM_MAX_TOKENS must be an integer",
    ):
        LLMSettings.from_env()


def test_llm_settings_reject_nonpositive_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")

    with pytest.raises(
        ConfigurationError,
        match="must be greater than zero",
    ):
        LLMSettings.from_env()