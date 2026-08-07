from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from invoice_system.documents import LoadedDocument
from invoice_system.models import Invoice, ValidationResult


ProcessingStage = Literal[
    "document_loading",
    "ingestion",
    "validation",
]

CurrentStage = Literal[
    "document_loading",
    "ingestion",
    "validation",
    "completed",
]

WorkflowStatus = Literal[
    "running",
    "completed",
    "failed",
]

AuditEventStatus = Literal[
    "succeeded",
    "failed",
]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class AuditEvent(BaseModel):
    """One observable event produced by a workflow stage."""

    model_config = ConfigDict(frozen=True)

    stage: ProcessingStage
    status: AuditEventStatus
    message: str
    duration_ms: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=utc_now)


class WorkflowError(BaseModel):
    """A technical failure that prevents workflow execution."""

    model_config = ConfigDict(frozen=True)

    stage: ProcessingStage
    error_type: str
    message: str
    timestamp: datetime = Field(default_factory=utc_now)


class WorkflowState(TypedDict, total=False):
    """
    Shared LangGraph state for one invoice-processing run.

    total=False allows nodes to return partial updates. A field is populated
    only after the stage responsible for that field succeeds.
    """

    invoice_path: str
    document: LoadedDocument
    invoice: Invoice
    validation_result: ValidationResult

    current_stage: CurrentStage
    status: WorkflowStatus

    audit_events: list[AuditEvent]
    errors: list[WorkflowError]