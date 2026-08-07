from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    ApprovalPolicyAssessment,
)
from invoice_system.database import initialize_database
from invoice_system.models import (
    Invoice,
    InvoiceItem,
    ValidationResult,
)
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


class FakeApprovalAgent:
    """Queue-based approval agent test double."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        *,
        prior_decision: ApprovalDecision | None = None,
        critique: ApprovalCritique | None = None,
    ) -> ApprovalDecision:
        self.calls.append(
            {
                "invoice": invoice,
                "validation_result": validation_result,
                "policy": policy,
                "prior_decision": prior_decision,
                "critique": critique,
            }
        )

        if not self.responses:
            raise AssertionError(
                "FakeApprovalAgent has no response left."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        if not isinstance(response, ApprovalDecision):
            raise AssertionError(
                "FakeApprovalAgent response must be ApprovalDecision."
            )

        return response


class FakeApprovalCritic:
    """Queue-based approval critic test double."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def review(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        decision: ApprovalDecision,
    ) -> ApprovalCritique:
        self.calls.append(
            {
                "invoice": invoice,
                "validation_result": validation_result,
                "policy": policy,
                "decision": decision,
            }
        )

        if not self.responses:
            raise AssertionError(
                "FakeApprovalCritic has no response left."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        if not isinstance(response, ApprovalCritique):
            raise AssertionError(
                "FakeApprovalCritic response must be ApprovalCritique."
            )

        return response


@pytest.fixture
def inventory_database(
    tmp_path: Path,
) -> Path:
    database_path = tmp_path / "inventory.db"
    initialize_database(database_path)
    return database_path


def make_valid_invoice(
    *,
    amount: str = "5000.00",
) -> Invoice:
    return Invoice(
        invoice_number="INV-TEST",
        vendor="Widgets Inc.",
        amount=Decimal(amount),
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


def make_invalid_invoice() -> Invoice:
    return Invoice(
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


def approve_decision(
    *,
    risk_level: str = "low",
    summary: str = "The invoice may be approved.",
) -> ApprovalDecision:
    return ApprovalDecision(
        decision="approve",
        risk_level=risk_level,  # type: ignore[arg-type]
        summary=summary,
        reasons=["The supplied controls support approval."],
    )


def reject_decision() -> ApprovalDecision:
    return ApprovalDecision(
        decision="reject",
        risk_level="high",
        summary="The invoice must be rejected.",
        reasons=["Deterministic validation failed."],
    )


def accept_critique() -> ApprovalCritique:
    return ApprovalCritique(
        verdict="accept",
        summary="The proposed decision is supported.",
        concerns=[],
        revision_instructions=[],
    )


def revise_critique(
    instruction: str = "Reconsider the supplied risk evidence.",
) -> ApprovalCritique:
    return ApprovalCritique(
        verdict="revise",
        summary="The decision needs revision.",
        concerns=["The rationale is not yet sufficient."],
        revision_instructions=[instruction],
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


def test_workflow_runs_through_approval_and_critic(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )
    approval_agent = FakeApprovalAgent(
        [approve_decision()]
    )
    approval_critic = FakeApprovalCritic(
        [accept_critique()]
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "completed"
    assert result["current_stage"] == "completed"
    assert result["validation_result"].passed is True
    assert result["approval_policy"].base_recommendation == "approve"
    assert result["approval_policy"].requires_additional_scrutiny is False
    assert result["approval_decision"].decision == "approve"
    assert result["approval_critique"].verdict == "accept"
    assert result["approval_revision_count"] == 0
    assert result["errors"] == []

    assert len(ingestion_agent.calls) == 1
    assert len(approval_agent.calls) == 1
    assert len(approval_critic.calls) == 1

    assert [
        event.stage
        for event in result["audit_events"]
    ] == [
        "document_loading",
        "ingestion",
        "validation",
        "approval_policy",
        "approval_decision",
        "approval_critique",
        "approval_finalization",
    ]


def test_high_value_critic_revision_returns_to_approval_agent(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    first_decision = approve_decision(
        risk_level="medium",
        summary="Initial high-value approval recommendation.",
    )
    corrected_decision = approve_decision(
        risk_level="high",
        summary="Revised high-value approval recommendation.",
    )
    first_critique = revise_critique(
        "Strengthen the risk rationale before approval."
    )
    final_critique = accept_critique()

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice(amount="15000.00")
    )
    approval_agent = FakeApprovalAgent(
        [first_decision, corrected_decision]
    )
    approval_critic = FakeApprovalCritic(
        [first_critique, final_critique]
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=inventory_database,
        max_approval_revisions=2,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "completed"
    assert result["approval_policy"].requires_additional_scrutiny is True
    assert result["approval_decision"] == corrected_decision
    assert result["approval_critique"] == final_critique
    assert result["approval_revision_count"] == 1

    assert len(approval_agent.calls) == 2
    assert len(approval_critic.calls) == 2

    revision_call = approval_agent.calls[1]
    assert revision_call["prior_decision"] == first_decision
    assert revision_call["critique"] == first_critique


def test_approval_loop_stops_at_manual_review_after_max_revisions(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    first_decision = approve_decision(risk_level="medium")
    revised_decision = approve_decision(risk_level="high")

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice(amount="15000.00")
    )
    approval_agent = FakeApprovalAgent(
        [first_decision, revised_decision]
    )
    approval_critic = FakeApprovalCritic(
        [revise_critique(), revise_critique()]
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=inventory_database,
        max_approval_revisions=1,
    )

    result = run_invoice_workflow(
        workflow,
        invoice_path,
    )

    assert result["status"] == "manual_review"
    assert result["current_stage"] == "manual_review"
    assert result["approval_revision_count"] == 1
    assert result["approval_critique"].verdict == "revise"
    assert result["errors"] == []

    assert len(approval_agent.calls) == 2
    assert len(approval_critic.calls) == 2
    assert result["audit_events"][-1].stage == "manual_review"


def test_business_validation_failure_flows_to_final_rejection(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_invalid_invoice()
    )
    approval_agent = FakeApprovalAgent(
        [reject_decision()]
    )
    approval_critic = FakeApprovalCritic(
        [accept_critique()]
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
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
    assert validation_result.issues[0].code == "insufficient_stock"

    policy = result["approval_policy"]
    assert policy.base_recommendation == "reject"
    assert policy.requires_additional_scrutiny is False

    assert result["approval_decision"].decision == "reject"
    assert result["approval_critique"].verdict == "accept"


def test_workflow_stops_after_document_loading_failure(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    missing_path = tmp_path / "missing.txt"

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )
    approval_agent = FakeApprovalAgent([])
    approval_critic = FakeApprovalCritic([])

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
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
    assert "approval_policy" not in result

    assert ingestion_agent.calls == []
    assert approval_agent.calls == []
    assert approval_critic.calls == []

    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "document_loading"
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
    approval_agent = FakeApprovalAgent([])
    approval_critic = FakeApprovalCritic([])

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
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
    assert approval_agent.calls == []
    assert approval_critic.calls == []

    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "ingestion"
    assert result["audit_events"][-1].status == "failed"


def test_validation_exception_fails_workflow(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )
    approval_agent = FakeApprovalAgent([])
    approval_critic = FakeApprovalCritic([])

    def failing_validator(
        invoice: Invoice,
        database_path: str | Path,
    ) -> ValidationResult:
        raise RuntimeError(
            "Simulated inventory database failure."
        )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
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
    assert approval_agent.calls == []
    assert approval_critic.calls == []

    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "validation"


def test_approval_agent_exception_fails_workflow(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )
    approval_agent = FakeApprovalAgent(
        [RuntimeError("Simulated approval model failure.")]
    )
    approval_critic = FakeApprovalCritic([])

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(workflow, invoice_path)

    assert result["status"] == "failed"
    assert result["current_stage"] == "approval_decision"
    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "approval_decision"
    assert approval_critic.calls == []


def test_approval_critic_exception_fails_workflow(
    tmp_path: Path,
    inventory_database: Path,
) -> None:
    invoice_path = write_invoice_file(tmp_path)

    ingestion_agent = FakeIngestionAgent(
        invoice=make_valid_invoice()
    )
    approval_agent = FakeApprovalAgent(
        [approve_decision()]
    )
    approval_critic = FakeApprovalCritic(
        [RuntimeError("Simulated critic model failure.")]
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=inventory_database,
    )

    result = run_invoice_workflow(workflow, invoice_path)

    assert result["status"] == "failed"
    assert result["current_stage"] == "approval_critique"
    assert len(result["errors"]) == 1
    assert result["errors"][0].stage == "approval_critique"
