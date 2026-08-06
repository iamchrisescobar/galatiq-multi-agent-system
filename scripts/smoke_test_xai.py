from pathlib import Path

from invoice_system.agents.ingestion import IngestionAgent
from invoice_system.config import LLMSettings
from invoice_system.llm import create_chat_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INVOICE = (
    PROJECT_ROOT
    / "data"
    / "invoices"
    / "invoice_1001.txt"
)


def main() -> None:
    settings = LLMSettings.from_env()

    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"Maximum output tokens: {settings.max_tokens}")

    model = create_chat_model(settings)

    invoice_text = SAMPLE_INVOICE.read_text(
        encoding="utf-8"
    )

    # Only one extraction attempt for the paid smoke test.
    agent = IngestionAgent(
        model,
        max_attempts=1,
    )

    invoice = agent.extract(invoice_text)

    print("\nStructured extraction:")
    print(invoice.model_dump_json(indent=2))


if __name__ == "__main__":
    main()