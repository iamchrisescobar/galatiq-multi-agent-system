from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from invoice_system.cli import (
    exit_code_for_result,
    format_cli_output,
)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _base_invoice():
    return _ns(
        invoice_number="INV-1001",
        vendor="Widgets Inc.",
        amount="5000.0",
    )


def test_default_output_for_paid_invoice_is_business_focused():
    result = {
        "status": "completed",
        "current_stage": "completed",
        "invoice": _base_invoice(),
        "validation_result": _ns(
            passed=True,
            issues=[],
        ),
        "approval_decision": _ns(
            decision="approve",
            risk_level="low",
            summary="Approved.",
        ),
        "payment_result": _ns(
            status="success",
            vendor="Widgets Inc.",
            amount="5000.0",
        ),
        "approval_revision_count": 0,
        "audit_events": [],
        "errors": [],
    }

    output = format_cli_output(
        result,
        elapsed_seconds=12.345,
    )

    assert "Outcome:   PAID" in output
    assert "Risk:      LOW" in output
    assert "Paid $5,000.00 to Widgets Inc." in output
    assert "Processing time: 12.35s" in output
    assert "Technical Details" not in output
    assert "Approval policy" not in output


def test_default_output_for_rejected_invoice_shows_reason_and_no_payment():
    result = {
        "status": "completed",
        "current_stage": "completed",
        "invoice": _ns(
            invoice_number="1002",
            vendor="Gadgets Co.",
            amount="15000.0",
        ),
        "validation_result": _ns(
            passed=False,
            issues=[
                _ns(
                    code="insufficient_stock",
                    message=(
                        "GadgetX requested quantity 20 "
                        "exceeds available stock 5."
                    ),
                    item="GadgetX",
                    requested_quantity=20,
                    available_stock=5,
                )
            ],
        ),
        "approval_decision": _ns(
            decision="reject",
            risk_level="high",
            summary=(
                "Invoice rejected due to failed "
                "deterministic validation."
            ),
        ),
        "rejection_record": _ns(
            status="rejected",
        ),
        "approval_revision_count": 0,
        "audit_events": [],
        "errors": [],
    }

    output = format_cli_output(
        result,
        rejection_log_path=Path(
            "data/rejections.jsonl"
        ),
    )

    assert "Outcome:   REJECTED" in output
    assert "Risk:      HIGH" in output
    assert (
        "GadgetX: requested 20, available 5"
        in output
    )
    assert (
        "Invoice rejected due to failed "
        "deterministic validation."
        in output
    )
    assert "Payment:   Not processed" in output
    assert "Record:    data" in output
    assert "rejections.jsonl" in output


def test_default_output_for_manual_review_shows_human_handoff():
    result = {
        "status": "manual_review",
        "current_stage": "manual_review",
        "invoice": _ns(
            invoice_number="INV-REVIEW",
            vendor="Review Vendor",
            amount="12500.0",
        ),
        "validation_result": _ns(
            passed=True,
            issues=[],
        ),
        "approval_decision": _ns(
            decision="approve",
            risk_level="high",
            summary="Proposed approval.",
        ),
        "manual_review_record": _ns(
            reason=(
                "Automated approval could not "
                "converge after 2 revisions."
            ),
        ),
        "approval_revision_count": 2,
        "audit_events": [],
        "errors": [],
    }

    output = format_cli_output(
        result,
        manual_review_log_path=Path(
            "data/manual_reviews.jsonl"
        ),
    )

    assert "Outcome:   MANUAL REVIEW" in output
    assert "Decision:   Pending human review" in output
    assert (
        "Automated approval could not "
        "converge after 2 revisions."
        in output
    )
    assert "Payment:   Not processed" in output
    assert "Queue:     data" in output
    assert "manual_reviews.jsonl" in output


def test_default_output_for_failed_workflow_shows_errors():
    result = {
        "status": "failed",
        "current_stage": "payment",
        "invoice": _base_invoice(),
        "validation_result": _ns(
            passed=True,
            issues=[],
        ),
        "approval_decision": _ns(
            decision="approve",
            risk_level="low",
            summary="Approved.",
        ),
        "approval_revision_count": 0,
        "audit_events": [],
        "errors": [
            _ns(
                stage="payment",
                error_type="RuntimeError",
                message="Payment service unavailable.",
            )
        ],
    }

    output = format_cli_output(result)

    assert "Outcome:   FAILED" in output
    assert "Stage:     payment" in output
    assert "RuntimeError" in output
    assert "Payment service unavailable." in output


def test_verbose_output_adds_technical_details_and_audit_events():
    result = {
        "status": "completed",
        "current_stage": "completed",
        "invoice": _base_invoice(),
        "validation_result": _ns(
            passed=True,
            issues=[],
        ),
        "approval_policy": {
            "base_recommendation": "approve"
        },
        "approval_decision": _ns(
            decision="approve",
            risk_level="low",
            summary="Approved.",
        ),
        "approval_critique": {
            "verdict": "accept"
        },
        "payment_result": _ns(
            status="success",
            vendor="Widgets Inc.",
            amount="5000.0",
        ),
        "approval_revision_count": 0,
        "audit_events": [
            _ns(
                stage="payment",
                status="succeeded",
                duration_ms=0.4,
                message="Payment completed.",
            )
        ],
        "errors": [],
    }

    output = format_cli_output(
        result,
        verbose=True,
        provider="xai",
        model="grok-build-0.1",
        invoice_path="data/invoices/invoice_1001.txt",
        captured_stdout="Paid 5000.0 to Widgets Inc.\n",
    )

    assert "Technical Details" in output
    assert "Provider: xai" in output
    assert "Model: grok-build-0.1" in output
    assert (
        "Invoice path: "
        "data/invoices/invoice_1001.txt"
        in output
    )
    assert "Approval policy" in output
    assert "Latest approval critique" in output
    assert "Audit events" in output
    assert "payment: succeeded" in output
    assert "Captured workflow stdout" in output
    assert "Paid 5000.0 to Widgets Inc." in output


def test_exit_codes_distinguish_completed_manual_review_and_failure():
    assert exit_code_for_result(
        {"status": "completed"}
    ) == 0
    assert exit_code_for_result(
        {"status": "manual_review"}
    ) == 2
    assert exit_code_for_result(
        {"status": "failed"}
    ) == 1
    assert exit_code_for_result(
        {"status": "unexpected"}
    ) == 1
