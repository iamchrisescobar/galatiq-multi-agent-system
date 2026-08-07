from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from invoice_system.agents.approval import (
    ApprovalAgent,
    ApprovalCritic,
)
from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    assess_approval_policy,
)
from invoice_system.models import (
    Invoice,
    InvoiceItem,
    ValidationIssue,
    ValidationResult,
)


class StubStructuredModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocation_count = 0
        self.received_messages: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.invocation_count += 1
        self.received_messages.append(list(messages))

        if not self.responses:
            raise AssertionError(
                "StubStructuredModel has no response left."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class StubChatModel:
    def __init__(self, responses: list[Any]) -> None:
        self.structured_model = StubStructuredModel(
            responses
        )
        self.requested_schema: type[Any] | None = None

    def with_structured_output(
        self,
        schema: type[Any],
    ) -> StubStructuredModel:
        self.requested_schema = schema
        return self.structured_model


def make_invoice(
    *,
    amount: str = "5000.00",
) -> Invoice:
    return Invoice(
        invoice_number="INV-APPROVAL-TEST",
        vendor="Widgets Inc.",
        amount=Decimal(amount),
        items=[
            InvoiceItem(
                name="WidgetA",
                quantity=5,
            )
        ],
        invoice_date=date(2026, 1, 15),
        due_date=date(2026, 2, 15),
    )


def make_passed_validation() -> ValidationResult:
    return ValidationResult(
        passed=True,
        issues=[],
    )


def make_failed_validation() -> ValidationResult:
    return ValidationResult(
        passed=False,
        issues=[
            ValidationIssue(
                code="insufficient_stock",
                message=(
                    "WidgetA requested quantity 20 exceeds "
                    "available stock 15."
                ),
                item="WidgetA",
                requested_quantity=20,
                available_stock=15,
            )
        ],
    )


def test_policy_recommends_approval_for_valid_low_value_invoice(
) -> None:
    invoice = make_invoice(amount="5000.00")
    validation = make_passed_validation()

    policy = assess_approval_policy(
        invoice,
        validation,
    )

    assert policy.base_recommendation == "approve"
    assert policy.amount_over_threshold is False
    assert policy.requires_additional_scrutiny is False
    assert policy.blocking_reasons == []
    assert policy.scrutiny_reasons == []


def test_policy_requires_additional_scrutiny_above_threshold(
) -> None:
    invoice = make_invoice(amount="15000.00")
    validation = make_passed_validation()

    policy = assess_approval_policy(
        invoice,
        validation,
    )

    assert policy.base_recommendation == "approve"
    assert policy.amount_over_threshold is True
    assert policy.requires_additional_scrutiny is True
    assert len(policy.scrutiny_reasons) == 1


def test_policy_recommends_rejection_after_validation_failure(
) -> None:
    invoice = make_invoice(amount="15000.00")
    validation = make_failed_validation()

    policy = assess_approval_policy(
        invoice,
        validation,
    )

    assert policy.base_recommendation == "reject"
    assert policy.validation_passed is False
    assert policy.validation_issue_codes == [
        "insufficient_stock"
    ]
    assert policy.amount_over_threshold is True
    assert policy.requires_additional_scrutiny is False
    assert len(policy.blocking_reasons) == 1
    assert policy.scrutiny_reasons == []


def test_approval_agent_returns_compliant_decision() -> None:
    expected = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="The invoice passed all configured controls.",
        reasons=[
            "Deterministic validation passed.",
            "The amount is below the scrutiny threshold.",
        ],
    )

    model = StubChatModel([expected])
    agent = ApprovalAgent(model)  # type: ignore[arg-type]

    invoice = make_invoice()
    validation = make_passed_validation()
    policy = assess_approval_policy(
        invoice,
        validation,
    )

    result = agent.decide(
        invoice,
        validation,
        policy,
    )

    assert result == expected
    assert model.requested_schema is ApprovalDecision
    assert model.structured_model.invocation_count == 1


def test_approval_agent_retries_validation_policy_violation(
) -> None:
    invalid_approval = ApprovalDecision(
        decision="approve",
        risk_level="medium",
        summary="Approve.",
        reasons=["No concerns identified."],
    )

    corrected_rejection = ApprovalDecision(
        decision="reject",
        risk_level="high",
        summary="The invoice failed deterministic validation.",
        reasons=[
            "The requested quantity exceeds available stock."
        ],
    )

    model = StubChatModel(
        [
            invalid_approval,
            corrected_rejection,
        ]
    )

    agent = ApprovalAgent(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    invoice = make_invoice()
    validation = make_failed_validation()
    policy = assess_approval_policy(
        invoice,
        validation,
    )

    result = agent.decide(
        invoice,
        validation,
        policy,
    )

    assert result == corrected_rejection
    assert model.structured_model.invocation_count == 2


def test_high_value_invoice_cannot_be_low_risk() -> None:
    invalid_decision = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="Approve.",
        reasons=["Validation passed."],
    )

    corrected_decision = ApprovalDecision(
        decision="approve",
        risk_level="medium",
        summary=(
            "The invoice passed validation but is high value."
        ),
        reasons=[
            "Validation passed.",
            "The amount exceeds the scrutiny threshold.",
        ],
    )

    model = StubChatModel(
        [
            invalid_decision,
            corrected_decision,
        ]
    )

    agent = ApprovalAgent(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    invoice = make_invoice(amount="15000.00")
    validation = make_passed_validation()
    policy = assess_approval_policy(
        invoice,
        validation,
    )

    result = agent.decide(
        invoice,
        validation,
        policy,
    )

    assert result == corrected_decision
    assert result.decision == "approve"
    assert result.risk_level == "medium"
    assert model.structured_model.invocation_count == 2


def test_revision_requires_prior_decision_and_critique_together(
) -> None:
    model = StubChatModel([])
    agent = ApprovalAgent(model)  # type: ignore[arg-type]

    invoice = make_invoice()
    validation = make_passed_validation()
    policy = assess_approval_policy(invoice, validation)

    prior_decision = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="Initial decision.",
        reasons=["Validation passed."],
    )

    with pytest.raises(ValueError):
        agent.decide(
            invoice,
            validation,
            policy,
            prior_decision=prior_decision,
        )


def test_approval_critic_returns_structured_review() -> None:
    expected = ApprovalCritique(
        verdict="accept",
        summary=(
            "The proposed approval is consistent with the "
            "available evidence."
        ),
        concerns=[],
        revision_instructions=[],
    )

    model = StubChatModel([expected])
    critic = ApprovalCritic(model)  # type: ignore[arg-type]

    invoice = make_invoice()
    validation = make_passed_validation()
    policy = assess_approval_policy(
        invoice,
        validation,
    )

    decision = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="The invoice passed all controls.",
        reasons=["Deterministic validation passed."],
    )

    result = critic.review(
        invoice,
        validation,
        policy,
        decision,
    )

    assert result == expected
    assert model.requested_schema is ApprovalCritique
    assert model.structured_model.invocation_count == 1


def test_approval_critic_retries_inconsistent_revision() -> None:
    invalid_critique = {
        "verdict": "revise",
        "summary": "The decision needs more work.",
        "concerns": ["The rationale is incomplete."],
        "revision_instructions": [],
    }

    corrected_critique = ApprovalCritique(
        verdict="revise",
        summary="The decision needs a clearer rationale.",
        concerns=["The rationale is incomplete."],
        revision_instructions=[
            "Explain how validation and invoice value support the decision."
        ],
    )

    model = StubChatModel(
        [
            invalid_critique,
            corrected_critique,
        ]
    )
    critic = ApprovalCritic(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    invoice = make_invoice()
    validation = make_passed_validation()
    policy = assess_approval_policy(invoice, validation)
    decision = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="Approve.",
        reasons=["Validation passed."],
    )

    result = critic.review(
        invoice,
        validation,
        policy,
        decision,
    )

    assert result == corrected_critique
    assert model.structured_model.invocation_count == 2


def test_critic_cannot_accept_hard_policy_violation() -> None:
    invalid_accept = ApprovalCritique(
        verdict="accept",
        summary="The decision is acceptable.",
        concerns=[],
        revision_instructions=[],
    )

    corrected_revision = ApprovalCritique(
        verdict="revise",
        summary="The risk classification violates policy.",
        concerns=[
            "A high-value invoice cannot be classified as low risk."
        ],
        revision_instructions=[
            "Reassess the invoice using medium or high risk."
        ],
    )

    model = StubChatModel(
        [
            invalid_accept,
            corrected_revision,
        ]
    )
    critic = ApprovalCritic(
        model,  # type: ignore[arg-type]
        max_attempts=2,
    )

    invoice = make_invoice(amount="15000.00")
    validation = make_passed_validation()
    policy = assess_approval_policy(invoice, validation)

    # Constructed directly to prove the critic also defends hard policy if a
    # bad decision ever reaches it through a future integration change.
    decision = ApprovalDecision(
        decision="approve",
        risk_level="low",
        summary="Approve.",
        reasons=["Validation passed."],
    )

    result = critic.review(
        invoice,
        validation,
        policy,
        decision,
    )

    assert result == corrected_revision
    assert model.structured_model.invocation_count == 2
