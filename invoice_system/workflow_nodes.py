from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from invoice_system.documents import (
    LoadedDocument,
    load_document,
)
from invoice_system.models import Invoice, ValidationResult
from invoice_system.validation import validate_invoice
from invoice_system.workflow_state import (
    AuditEvent,
    ProcessingStage,
    WorkflowError,
    WorkflowState,
)


class InvoiceExtractor(Protocol):
    """
    Interface required from an ingestion agent.

    IngestionAgent already satisfies this protocol because it provides
    extract(document_text) -> Invoice.
    """

    def extract(self, document_text: str) -> Invoice:
        ...


DocumentLoader = Callable[[str | Path], LoadedDocument]

InvoiceValidator = Callable[
    [Invoice, str | Path],
    ValidationResult,
]


@dataclass(frozen=True)
class WorkflowDependencies:
    """Dependencies injected into workflow nodes."""

    ingestion_agent: InvoiceExtractor
    database_path: str | Path
    document_loader: DocumentLoader = load_document
    invoice_validator: InvoiceValidator = validate_invoice


class WorkflowNodes:
    """Deterministic node implementations for the invoice workflow."""

    def __init__(
        self,
        dependencies: WorkflowDependencies,
    ) -> None:
        self._dependencies = dependencies

    def load_document(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Load the invoice file into normalized document text."""

        started_at = perf_counter()

        try:
            invoice_path = state.get("invoice_path")

            if not invoice_path:
                raise ValueError(
                    "Workflow state does not contain an invoice path."
                )

            document = self._dependencies.document_loader(
                invoice_path
            )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="document_loading",
                started_at=started_at,
                exc=exc,
            )

        event = AuditEvent(
            stage="document_loading",
            status="succeeded",
            message=(
                f"Loaded invoice document "
                f"{document.path.name} as {document.media_type}."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "document": document,
            "current_stage": "document_loading",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def ingest_invoice(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Extract a typed invoice from the loaded document."""

        started_at = perf_counter()

        try:
            document = state.get("document")

            if document is None:
                raise ValueError(
                    "Workflow state does not contain a loaded document."
                )

            invoice = self._dependencies.ingestion_agent.extract(
                document.text
            )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="ingestion",
                started_at=started_at,
                exc=exc,
            )

        invoice_identifier = (
            invoice.invoice_number
            or "unknown invoice number"
        )

        event = AuditEvent(
            stage="ingestion",
            status="succeeded",
            message=(
                f"Extracted structured data for "
                f"{invoice_identifier}."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "invoice": invoice,
            "current_stage": "ingestion",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def validate_invoice(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Validate the extracted invoice against business rules."""

        started_at = perf_counter()

        try:
            invoice = state.get("invoice")

            if invoice is None:
                raise ValueError(
                    "Workflow state does not contain an invoice."
                )

            validation_result = (
                self._dependencies.invoice_validator(
                    invoice,
                    self._dependencies.database_path,
                )
            )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="validation",
                started_at=started_at,
                exc=exc,
            )

        issue_count = len(validation_result.issues)

        event = AuditEvent(
            stage="validation",
            status="succeeded",
            message=(
                "Validation completed with "
                f"{issue_count} issue(s)."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "validation_result": validation_result,
            "current_stage": "completed",
            "status": "completed",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def _failure_update(
        self,
        *,
        state: WorkflowState,
        stage: ProcessingStage,
        started_at: float,
        exc: Exception,
    ) -> WorkflowState:
        """Create a consistent state update for node failures."""

        workflow_error = WorkflowError(
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
        )

        event = AuditEvent(
            stage=stage,
            status="failed",
            message=(
                f"{stage.replace('_', ' ').title()} failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "current_stage": stage,
            "status": "failed",
            "errors": [
                *state.get("errors", []),
                workflow_error,
            ],
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """Return elapsed execution time in milliseconds."""

        return round(
            (perf_counter() - started_at) * 1000,
            3,
        )