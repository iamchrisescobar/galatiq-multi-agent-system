from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_system.database import initialize_database
from invoice_system.models import Invoice, InvoiceItem
from invoice_system.workflow import (
    build_invoice_workflow,
    run_invoice_workflow,
)


class FakeIngestionAgent:
    """Test double that avoids all paid LLM calls."""

    def __init__(
        self,
        *,
        invoice: Invoice | None = None,
        error: Exception | None = None,
    ) -> None:
        self._invoice = invoice
        self._error = error
        self.calls: list[str] = []

    def extract(
        self,
        document_text: str,
    ) -> Invoice:
        self.calls.append(document_text)

        if self._error is not None:
            raise self._error

        if self._invoice is None:
            raise AssertionError(
                "FakeIngestionAgent was not given an invoice."
            )

        return self._invoice


@pytest.fixture
def inventory_database(
    tmp_path: Path,
) -> Path:
    database_path = tmp_path / "inventory.db"
    initialize_database(database_path)
    return database_path


def make_valid_invoice() -> Invoice:
    return Invoice(
        invoice_number="INV-TEST",
        vendor="Widgets Inc.",
        amount=Decimal("5000.00"),
        items=[
            InvoiceItem(
                name="WidgetA",
                quantity=10,
            ),
            InvoiceItem(
                name="WidgetB",
                quantity=5,
            ),
        ],
        invoice_date=date(2026, 1, 15),
        due_date=date(2026, 2, 1),
    )


def write_invoice_file(
    tmp_path: Path,
) -> Path:
    invoice_path = tmp_path / "invoice.txt"
    invoice_path.write_text(
        (
            "INVOICE\n"
            "Vendor: Widgets Inc.\n"
            "Invoice Number: INV-TEST\n"
            "Total: $5,000.00\n"
        ),
        encoding="utf-8",
    )
    return invoice_path


def test_workflow_runs_through_validation(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "completed"
    assert result["current_stage"] == "completed"
    assert result["validation_result"].passed is True
    assert result["errors"] == []

    assert len(ingestion_agent.calls) == 1
    assert "Widgets Inc." in ingestion_agent.calls[0]

    assert [
        event.stage
        for event in result["audit_events"]
    ] == [
        "document_loading",
        "ingestion",
        "validation",
    ]

    assert all(
        event.status == "succeeded"
        for event in result["audit_events"]
    )


def test_workflow_stops_after_document_loading_failure(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    missing_path = tmp_path / "missing.txt"

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(
        workflow,
        missing_path,
    )

    assert result["status"] == "failed"
    assert result["current_stage"] == "document_loading"

    assert "document" not in result
    assert "invoice" not in result
    assert "validation_result" not in result

    assert ingestion_agent.calls == []

    assert len(result["errors"]) == 1
    assert (
        result["errors"][0].stage
        == "document_loading"
    )

    assert len(result["audit_events"]) == 1
    assert result["audit_events"][0].status == "failed"


def test_workflow_stops_after_ingestion_failure(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        error=RuntimeError(
            "Simulated structured extraction failure."
        )
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "failed"
    assert result["current_stage"] == "ingestion"

    assert "document" in result
    assert "invoice" not in result
    assert "validation_result" not in result

    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "ingestion"

    assert [
        event.stage
        for event in result["audit_events"]
    ] == [
        "document_loading",
        "ingestion",
    ]

    assert result["audit_events"][-1].status == "failed"


def test_business_validation_failure_completes_workflow(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    invalid_invoice = Invoice(
        invoice_number="INV-1002",
        vendor="Gadgets Co.",
        amount=Decimal("15000.00"),
        items=[
            InvoiceItem(
                name="GadgetX",
                quantity=20,
            )
        ],
        invoice_date=date(2026, 1, 30),
        due_date=date(2026, 2, 28),
    )

    ingestion_agent = FakeIngestionAgent(
        invoice=invalid_invoice
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "completed"
    assert result["current_stage"] == "completed"
    assert result["errors"] == []

    validation_result = result["validation_result"]

    assert validation_result.passed is False
    assert len(validation_result.issues) == 1
    assert (
        validation_result.issues[0].code
        == "insufficient_stock"
    )


def test_validation_exception_fails_workflow(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )

    def failing_validator(
        invoice: Invoice,
        database_path: str | Path,
    ):
        raise RuntimeError(
            "Simulated inventory database failure."
        )

    workflow = build_invoice_workflow(
        ingestion_agent,
        database_path=inventory_database,
        invoice_validator=failing_validator,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "failed"
    assert result["current_stage"] == "validation"

    assert "invoice" in result
    assert "validation_result" not in result

    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "validation"