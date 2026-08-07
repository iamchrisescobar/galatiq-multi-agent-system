from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    ApprovalPolicyAssessment,
)
from invoice_system.models import Invoice, ValidationResult


DEFAULT_REJECTION_LOG_PATH = Path("data/rejections.jsonl")
DEFAULT_MANUAL_REVIEW_LOG_PATH = Path("data/manual_reviews.jsonl")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class RejectionRecord(BaseModel):
    """Durable record for a final automated rejection."""

    model_config = ConfigDict(frozen=True)

    status: Literal["rejected"] = "rejected"
    invoice: Invoice
    validation_result: ValidationResult
    approval_decision: ApprovalDecision
    approval_critique: ApprovalCritique
    created_at: datetime = Field(default_factory=utc_now)


class ManualReviewRecord(BaseModel):
    """Durable work item for an invoice requiring human review."""

    model_config = ConfigDict(frozen=True)

    status: Literal["pending"] = "pending"
    reason: str = Field(min_length=1)
    revision_count: int = Field(ge=0)

    invoice: Invoice
    validation_result: ValidationResult
    approval_policy: ApprovalPolicyAssessment
    proposed_decision: ApprovalDecision
    latest_critique: ApprovalCritique

    created_at: datetime = Field(default_factory=utc_now)


def _append_jsonl(
    record: BaseModel,
    log_path: str | Path,
) -> None:
    """Append one Pydantic model as a JSON Lines record."""

    path = Path(log_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


def log_rejection(
    invoice: Invoice,
    validation_result: ValidationResult,
    decision: ApprovalDecision,
    critique: ApprovalCritique,
    log_path: str | Path = DEFAULT_REJECTION_LOG_PATH,
) -> RejectionRecord:
    """Persist a final rejection and the reasoning that produced it."""

    if decision.decision != "reject":
        raise ValueError(
            "Only a finalized reject decision may be logged as a rejection."
        )

    if critique.verdict != "accept":
        raise ValueError(
            "A rejection may only be logged after critic acceptance."
        )

    record = RejectionRecord(
        invoice=invoice,
        validation_result=validation_result,
        approval_decision=decision,
        approval_critique=critique,
    )

    _append_jsonl(
        record,
        log_path,
    )

    return record


def log_manual_review(
    invoice: Invoice,
    validation_result: ValidationResult,
    policy: ApprovalPolicyAssessment,
    decision: ApprovalDecision,
    critique: ApprovalCritique,
    revision_count: int,
    reason: str,
    log_path: str | Path = DEFAULT_MANUAL_REVIEW_LOG_PATH,
) -> ManualReviewRecord:
    """Persist a pending work item for human review."""

    if critique.verdict != "revise":
        raise ValueError(
            "Manual review requires an unresolved revise critique."
        )

    record = ManualReviewRecord(
        reason=reason,
        revision_count=revision_count,
        invoice=invoice,
        validation_result=validation_result,
        approval_policy=policy,
        proposed_decision=decision,
        latest_critique=critique,
    )

    _append_jsonl(
        record,
        log_path,
    )

    return record
