from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from invoice_system.models import (
    Invoice,
    ValidationIssueCode,
    ValidationResult,
)


DEFAULT_SCRUTINY_THRESHOLD = Decimal("10000.00")


ApprovalRecommendation = Literal[
    "approve",
    "reject",
]

ApprovalDecisionValue = Literal[
    "approve",
    "reject",
]

ApprovalRiskLevel = Literal[
    "low",
    "medium",
    "high",
]

ApprovalCritiqueVerdict = Literal[
    "accept",
    "revise",
]


class ApprovalPolicyAssessment(BaseModel):
    """Deterministic policy facts supplied to the approval agents."""

    model_config = ConfigDict(frozen=True)

    scrutiny_threshold: Decimal
    invoice_amount: Decimal | None

    validation_passed: bool
    validation_issue_codes: list[ValidationIssueCode] = Field(
        default_factory=list
    )

    amount_over_threshold: bool
    requires_additional_scrutiny: bool

    base_recommendation: ApprovalRecommendation
    blocking_reasons: list[str] = Field(default_factory=list)
    scrutiny_reasons: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    """Structured VP-level approve/reject recommendation."""

    model_config = ConfigDict(frozen=True)

    decision: ApprovalDecisionValue
    risk_level: ApprovalRiskLevel
    summary: str = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)


class ApprovalCritique(BaseModel):
    """Independent review of an approval decision."""

    model_config = ConfigDict(frozen=True)

    verdict: ApprovalCritiqueVerdict
    summary: str = Field(min_length=1)
    concerns: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(
        default_factory=list
    )


def assess_approval_policy(
    invoice: Invoice,
    validation_result: ValidationResult,
    *,
    scrutiny_threshold: Decimal = DEFAULT_SCRUTINY_THRESHOLD,
) -> ApprovalPolicyAssessment:
    """
    Derive deterministic approval-policy facts.

    The LLM may reason about and explain these facts, but it may not override
    them. A high-value invoice can still be recommended for approval; the
    additional-scrutiny flag tells the workflow that the recommendation must
    survive the independent critic before it can become final.
    """

    if scrutiny_threshold <= 0:
        raise ValueError(
            "scrutiny_threshold must be greater than zero."
        )

    invoice_amount = invoice.amount

    amount_over_threshold = (
        invoice_amount is not None
        and invoice_amount > scrutiny_threshold
    )

    blocking_reasons = [
        issue.message
        for issue in validation_result.issues
    ]

    # Validation failures are already deterministic blockers. They should be
    # rejected rather than held in a separate high-value review state.
    requires_additional_scrutiny = (
        validation_result.passed
        and amount_over_threshold
    )

    scrutiny_reasons: list[str] = []

    if requires_additional_scrutiny:
        scrutiny_reasons.append(
            (
                f"Invoice amount {invoice_amount} exceeds the "
                f"additional-scrutiny threshold of "
                f"{scrutiny_threshold}."
            )
        )

    base_recommendation: ApprovalRecommendation = (
        "approve"
        if validation_result.passed
        else "reject"
    )

    return ApprovalPolicyAssessment(
        scrutiny_threshold=scrutiny_threshold,
        invoice_amount=invoice_amount,
        validation_passed=validation_result.passed,
        validation_issue_codes=[
            issue.code
            for issue in validation_result.issues
        ],
        amount_over_threshold=amount_over_threshold,
        requires_additional_scrutiny=requires_additional_scrutiny,
        base_recommendation=base_recommendation,
        blocking_reasons=blocking_reasons,
        scrutiny_reasons=scrutiny_reasons,
    )


def approval_decision_policy_violations(
    decision: ApprovalDecision,
    policy: ApprovalPolicyAssessment,
) -> list[str]:
    """Return hard-policy violations in a proposed approval decision."""

    violations: list[str] = []

    if (
        not policy.validation_passed
        and decision.decision != "reject"
    ):
        violations.append(
            (
                "an invoice that failed deterministic "
                "validation must be rejected"
            )
        )

    if (
        policy.amount_over_threshold
        and decision.risk_level == "low"
    ):
        violations.append(
            (
                "an invoice above the scrutiny threshold "
                "cannot be classified as low risk"
            )
        )

    return violations
