from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    ApprovalPolicyAssessment,
)
from invoice_system.models import (
    Invoice,
    InvoiceItem,
    ValidationIssue,
    ValidationResult,
)
from invoice_system.outcome_records import (
    log_manual_review,
    log_rejection,
)


def make_invoice() -> Invoice:
    return Invoice(
        invoice_number="INV-TEST",
        vendor="Widgets Inc.",
        amount=Decimal("15000.00"),
        items=[
            InvoiceItem(
                name="WidgetA",
                quantity=10,
            )
        ],
        invoice_date=date(2026, 1, 15),
        due_date=date(2026, 2, 1),
    )


def make_failed_validation() -> ValidationResult:
    return ValidationResult(
        passed=False,
        issues=[
            ValidationIssue(
                code="insufficient_stock",
                message="Requested quantity exceeds available stock.",
                item="WidgetA",
                requested_quantity=20,
                available_stock=15,
            )
        ],
    )


def make_passed_validation() -> ValidationResult:
    return ValidationResult(
        passed=True,
        issues=[],
    )


def make_reject_decision() -> ApprovalDecision:
    return ApprovalDecision(
        decision="reject",
        risk_level="high",
        summary="Reject because deterministic validation failed.",
        reasons=["Insufficient stock."],
    )


def make_approve_decision() -> ApprovalDecision:
    return ApprovalDecision(
        decision="approve",
        risk_level="high",
        summary="Approve after review.",
        reasons=["Validation passed."],
    )


def make_policy() -> ApprovalPolicyAssessment:
    return ApprovalPolicyAssessment(
        scrutiny_threshold=Decimal("10000.00"),
        invoice_amount=Decimal("15000.00"),
        validation_passed=True,
        validation_issue_codes=[],
        amount_over_threshold=True,
        requires_additional_scrutiny=True,
        base_recommendation="approve",
        blocking_reasons=[],
        scrutiny_reasons=[
            "Invoice amount exceeds the additional-scrutiny threshold."
        ],
    )




def make_accept_critique() -> ApprovalCritique:
    return ApprovalCritique(
        verdict="accept",
        summary="The proposed rejection is supported.",
        concerns=[],
        revision_instructions=[],
    )

def make_revise_critique() -> ApprovalCritique:
    return ApprovalCritique(
        verdict="revise",
        summary="The decision still needs revision.",
        concerns=["Risk rationale remains incomplete."],
        revision_instructions=["Address the identified risk."],
    )


def test_log_rejection_appends_jsonl_record(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "rejections.jsonl"

    record = log_rejection(
        make_invoice(),
        make_failed_validation(),
        make_reject_decision(),
        make_accept_critique(),
        log_path,
    )

    assert record.status == "rejected"
    assert record.approval_decision.decision == "reject"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["status"] == "rejected"
    assert payload["invoice"]["invoice_number"] == "INV-TEST"
    assert payload["approval_decision"]["decision"] == "reject"
    assert payload["approval_critique"]["verdict"] == "accept"
    assert payload["validation_result"]["passed"] is False


def test_log_rejection_refuses_approve_decision(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Only a finalized reject decision",
    ):
        log_rejection(
            make_invoice(),
            make_passed_validation(),
            make_approve_decision(),
            make_accept_critique(),
            tmp_path / "rejections.jsonl",
        )


def test_log_manual_review_creates_pending_work_item(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "manual_reviews.jsonl"
    reason = (
        "Approval could not converge after 2 revisions; "
        "human review is required."
    )

    record = log_manual_review(
        make_invoice(),
        make_passed_validation(),
        make_policy(),
        make_approve_decision(),
        make_revise_critique(),
        2,
        reason,
        log_path,
    )

    assert record.status == "pending"
    assert record.revision_count == 2
    assert record.latest_critique.verdict == "revise"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["status"] == "pending"
    assert payload["invoice"]["invoice_number"] == "INV-TEST"
    assert payload["revision_count"] == 2
    assert payload["latest_critique"]["verdict"] == "revise"


def test_log_manual_review_requires_unresolved_revision(
    tmp_path: Path,
) -> None:
    accepted_critique = ApprovalCritique(
        verdict="accept",
        summary="Decision is acceptable.",
        concerns=[],
        revision_instructions=[],
    )

    with pytest.raises(
        ValueError,
        match="Manual review requires an unresolved revise critique",
    ):
        log_manual_review(
            make_invoice(),
            make_passed_validation(),
            make_policy(),
            make_approve_decision(),
            accepted_critique,
            2,
            "Human review required.",
            tmp_path / "manual_reviews.jsonl",
        )
