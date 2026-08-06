from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from invoice_system.agents.ingestion import (
    IngestionAgent,
    InvoiceExtractionError,
)
from invoice_system.models import Invoice, InvoiceItem


class StubStructuredModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocation_count = 0
        self.received_messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.invocation_count += 1
        self.received_messages.append(messages)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class StubChatModel:
    def __init__(self, responses: list[Any]) -> None:
        self.structured_model = StubStructuredModel(responses)
        self.requested_schema: type[Any] | None = None

    def with_structured_output(
        self,
        schema: type[Any],
    ) -> StubStructuredModel:
        self.requested_schema = schema
        return self.structured_model


def test_ingestion_agent_returns_structured_invoice() -> None:
    expected = Invoice(
        invoice_number="INV-1001",
        vendor="Northstar Supplies",
        amount=Decimal("2500.00"),
        invoice_date=date(2026, 1, 1),
        due_date=date(2026, 1, 31),
        items=[
            InvoiceItem(name="WidgetA", quantity=5),
        ],
    )
    model = StubChatModel([expected])
    agent = IngestionAgent(model)  # type: ignore[arg-type]

    result = agent.extract("Example invoice document")

    assert result == expected
    assert model.requested_schema is Invoice
    assert model.structured_model.invocation_count == 1


def test_ingestion_agent_accepts_dictionary_response() -> None:
    model = StubChatModel(
        [
            {
                "invoice_number": "INV-1002",
                "vendor": "Example Vendor",
                "amount": "725.50",
                "invoice_date": "2026-02-01",
                "due_date": "2026-03-01",
                "items": [
                    {
                        "name": "GadgetX",
                        "quantity": 1,
                    }
                ],
            }
        ]
    )
    agent = IngestionAgent(model)  # type: ignore[arg-type]

    result = agent.extract("Example invoice document")

    assert result.invoice_number == "INV-1002"
    assert result.amount == Decimal("725.50")
    assert result.invoice_date == date(2026, 2, 1)


def test_ingestion_agent_retries_once_after_model_failure() -> None:
    expected = Invoice(
        invoice_number="INV-RETRY",
        vendor="Retry Vendor",
        amount=Decimal("100.00"),
        invoice_date=date(2026, 1, 1),
        due_date=date(2026, 1, 15),
        items=[InvoiceItem(name="WidgetA", quantity=1)],
    )

    model = StubChatModel(
        [
            ValueError("Invalid structured response"),
            expected,
        ]
    )
    agent = IngestionAgent(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    result = agent.extract("Example invoice document")

    assert result == expected
    assert model.structured_model.invocation_count == 2


def test_ingestion_agent_fails_after_attempt_limit() -> None:
    model = StubChatModel(
        [
            ValueError("First failure"),
            ValueError("Second failure"),
        ]
    )
    agent = IngestionAgent(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    with pytest.raises(
        InvoiceExtractionError,
        match="failed after 2 attempts",
    ):
        agent.extract("Example invoice document")

    assert model.structured_model.invocation_count == 2


def test_ingestion_agent_rejects_empty_document() -> None:
    model = StubChatModel([])
    agent = IngestionAgent(model)  # type: ignore[arg-type]

    with pytest.raises(
        InvoiceExtractionError,
        match="empty document",
    ):
        agent.extract("   ")

    assert model.structured_model.invocation_count == 0