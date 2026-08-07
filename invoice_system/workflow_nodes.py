from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    ApprovalPolicyAssessment,
    assess_approval_policy,
)
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
    """Interface required from an ingestion agent."""

    def extract(self, document_text: str) -> Invoice:
        ...


class ApprovalDecider(Protocol):
    """Interface required from the approval agent."""

    def decide(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        *,
        prior_decision: ApprovalDecision | None = None,
        critique: ApprovalCritique | None = None,
    ) -> ApprovalDecision:
        ...


class ApprovalReviewer(Protocol):
    """Interface required from the independent approval critic."""

    def review(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        decision: ApprovalDecision,
    ) -> ApprovalCritique:
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
    approval_agent: ApprovalDecider
    approval_critic: ApprovalReviewer
    database_path: str | Path
    document_loader: DocumentLoader = load_document
    invoice_validator: InvoiceValidator = validate_invoice


class WorkflowNodes:
    """Node implementations for the invoice workflow."""

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
            "current_stage": "validation",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def assess_approval(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Derive deterministic approval policy facts."""

        started_at = perf_counter()

        try:
            invoice = state.get("invoice")
            validation_result = state.get("validation_result")

            if invoice is None:
                raise ValueError(
                    "Workflow state does not contain an invoice."
                )

            if validation_result is None:
                raise ValueError(
                    "Workflow state does not contain a validation result."
                )

            policy = assess_approval_policy(
                invoice,
                validation_result,
            )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="approval_policy",
                started_at=started_at,
                exc=exc,
            )

        event = AuditEvent(
            stage="approval_policy",
            status="succeeded",
            message=(
                "Approval policy assessed. "
                f"Base recommendation: {policy.base_recommendation}; "
                "additional scrutiny required: "
                f"{policy.requires_additional_scrutiny}."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "approval_policy": policy,
            "current_stage": "approval_policy",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def make_approval_decision(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Produce an initial or critic-directed approval decision."""

        started_at = perf_counter()

        try:
            invoice = state.get("invoice")
            validation_result = state.get("validation_result")
            policy = state.get("approval_policy")

            if invoice is None:
                raise ValueError(
                    "Workflow state does not contain an invoice."
                )

            if validation_result is None:
                raise ValueError(
                    "Workflow state does not contain a validation result."
                )

            if policy is None:
                raise ValueError(
                    "Workflow state does not contain an approval policy."
                )

            prior_decision = state.get("approval_decision")
            critique = state.get("approval_critique")

            is_revision = (
                prior_decision is not None
                and critique is not None
                and critique.verdict == "revise"
            )

            decision = self._dependencies.approval_agent.decide(
                invoice,
                validation_result,
                policy,
                prior_decision=(
                    prior_decision
                    if is_revision
                    else None
                ),
                critique=(
                    critique
                    if is_revision
                    else None
                ),
            )

            revision_count = state.get(
                "approval_revision_count",
                0,
            )

            if is_revision:
                revision_count += 1

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="approval_decision",
                started_at=started_at,
                exc=exc,
            )

        event = AuditEvent(
            stage="approval_decision",
            status="succeeded",
            message=(
                f"Approval agent proposed {decision.decision} "
                f"at {decision.risk_level} risk."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "approval_decision": decision,
            "approval_revision_count": revision_count,
            "current_stage": "approval_decision",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def review_approval_decision(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Have the independent critic accept or challenge the decision."""

        started_at = perf_counter()

        try:
            invoice = state.get("invoice")
            validation_result = state.get("validation_result")
            policy = state.get("approval_policy")
            decision = state.get("approval_decision")

            if invoice is None:
                raise ValueError(
                    "Workflow state does not contain an invoice."
                )

            if validation_result is None:
                raise ValueError(
                    "Workflow state does not contain a validation result."
                )

            if policy is None:
                raise ValueError(
                    "Workflow state does not contain an approval policy."
                )

            if decision is None:
                raise ValueError(
                    "Workflow state does not contain an approval decision."
                )

            critique = self._dependencies.approval_critic.review(
                invoice,
                validation_result,
                policy,
                decision,
            )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="approval_critique",
                started_at=started_at,
                exc=exc,
            )

        event = AuditEvent(
            stage="approval_critique",
            status="succeeded",
            message=(
                f"Approval critic verdict: {critique.verdict}."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "approval_critique": critique,
            "current_stage": "approval_critique",
            "status": "running",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def finalize_approval(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Finalize a decision only after the critic has accepted it."""

        started_at = perf_counter()

        try:
            decision = state.get("approval_decision")
            critique = state.get("approval_critique")

            if decision is None:
                raise ValueError(
                    "Workflow state does not contain an approval decision."
                )

            if critique is None:
                raise ValueError(
                    "Workflow state does not contain an approval critique."
                )

            if critique.verdict != "accept":
                raise ValueError(
                    "Cannot finalize a decision the critic did not accept."
                )

        except Exception as exc:
            return self._failure_update(
                state=state,
                stage="approval_finalization",
                started_at=started_at,
                exc=exc,
            )

        event = AuditEvent(
            stage="approval_finalization",
            status="succeeded",
            message=(
                f"Final approval decision: {decision.decision}."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "current_stage": "completed",
            "status": "completed",
            "audit_events": [
                *state.get("audit_events", []),
                event,
            ],
        }

    def escalate_manual_review(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Stop safely when the bounded reflection loop cannot converge."""

        started_at = perf_counter()
        revision_count = state.get(
            "approval_revision_count",
            0,
        )

        event = AuditEvent(
            stage="manual_review",
            status="succeeded",
            message=(
                "Approval could not converge after "
                f"{revision_count} revision(s); escalated for "
                "manual review."
            ),
            duration_ms=self._elapsed_ms(started_at),
        )

        return {
            "current_stage": "manual_review",
            "status": "manual_review",
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
