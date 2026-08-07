from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import ValidationError

from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    ApprovalPolicyAssessment,
    approval_decision_policy_violations,
)
from invoice_system.models import Invoice, ValidationResult


APPROVAL_SYSTEM_PROMPT = """
You are the VP approval agent for an accounts-payable workflow.

Evaluate the supplied invoice using only the structured invoice,
deterministic validation result, approval-policy assessment, and any critic
feedback supplied for a revision.

Your job is to propose one of two decisions:
- approve
- reject

Hard controls:
- An invoice that failed deterministic validation must be rejected.
- Deterministic validation and policy facts cannot be overridden.
- A policy base recommendation of "approve" means there is no deterministic
  blocker; it does not force you to approve if the supplied facts justify a
  rejection.
- An invoice above the configured scrutiny threshold cannot be classified as
  low risk.
- A high-value invoice may receive an approve recommendation, but that
  recommendation is not final until the independent critic completes the
  required scrutiny and accepts it.
- Do not invent purchase orders, vendor history, contracts, delivery records,
  prior approvals, completed reviews, or other facts that were not supplied.
- Clearly distinguish verified facts from limitations in the available data.
- Give a concise, business-readable explanation.

When critic feedback is supplied, explicitly address the critic's concerns
and return a corrected decision.
""".strip()


CRITIC_SYSTEM_PROMPT = """
You are the independent approval critic for an accounts-payable workflow.

Review the proposed VP approval decision for:
- policy compliance
- factual grounding
- internal consistency
- financial risk
- unsupported assumptions

Your verdict must be one of:
- accept: the proposed approve/reject decision may become final
- revise: the approval agent must reconsider the decision using your feedback

Hard controls:
- An invoice that failed deterministic validation must be rejected.
- An invoice above the scrutiny threshold cannot be classified as low risk.
- When requires_additional_scrutiny is true, your review is the required
  high-value scrutiny gate. Apply heightened scrutiny before accepting the
  proposed decision.
- Do not invent facts that are not present in the supplied inputs.

Request revision when:
- The decision conflicts with deterministic validation or policy.
- The risk classification is inconsistent with the policy.
- The reasoning relies on unsupported facts.
- The reasoning does not justify the decision.
- Important supplied risks were ignored.

Accept only when the decision is safe, grounded, internally consistent, and
supported by the supplied evidence. When requesting revision, provide
specific revision instructions that the approval agent can act on.
""".strip()


class ApprovalDecisionError(RuntimeError):
    """Raised when a compliant approval decision cannot be produced."""


class ApprovalCritiqueError(RuntimeError):
    """Raised when a compliant approval critique cannot be produced."""


class ApprovalAgent:
    """Produce policy-constrained VP approval decisions."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self._structured_model = model.with_structured_output(
            ApprovalDecision
        )
        self._max_attempts = max_attempts

    def decide(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        *,
        prior_decision: ApprovalDecision | None = None,
        critique: ApprovalCritique | None = None,
    ) -> ApprovalDecision:
        """Produce an initial or revised approval decision."""

        if (prior_decision is None) != (critique is None):
            raise ValueError(
                "prior_decision and critique must be supplied together."
            )

        if critique is not None and critique.verdict != "revise":
            raise ValueError(
                "Only a revise critique may be used to request a new "
                "approval decision."
            )

        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=APPROVAL_SYSTEM_PROMPT),
            HumanMessage(
                content=self._build_request(
                    invoice=invoice,
                    validation_result=validation_result,
                    policy=policy,
                    prior_decision=prior_decision,
                    critique=critique,
                )
            ),
        ]

        errors: list[str] = []

        # This is a narrow model/schema/policy retry loop. The independent
        # approval <-> critic reflection loop is owned by LangGraph.
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._structured_model.invoke(
                    messages
                )

                decision = self._coerce_decision(response)

                policy_violations = (
                    approval_decision_policy_violations(
                        decision,
                        policy,
                    )
                )

                if policy_violations:
                    raise ApprovalDecisionError(
                        "; ".join(policy_violations)
                    )

                return decision

            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_message)

                if attempt == self._max_attempts:
                    break

                messages.append(
                    HumanMessage(
                        content=(
                            "The previous proposed decision did not "
                            "satisfy the approval schema or hard policy "
                            "controls.\n\n"
                            f"Problems: {error_message}\n\n"
                            "Return a corrected decision grounded only "
                            "in the supplied invoice, validation, policy "
                            "facts, and critic feedback if present."
                        )
                    )
                )

        raise ApprovalDecisionError(
            (
                "Approval decision failed after "
                f"{self._max_attempts} attempts: "
                f"{' | '.join(errors)}"
            )
        )

    @staticmethod
    def _build_request(
        *,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        prior_decision: ApprovalDecision | None,
        critique: ApprovalCritique | None,
    ) -> str:
        revision_context = (
            "This is the initial approval decision."
        )

        if prior_decision is not None and critique is not None:
            revision_context = (
                "Revise the prior decision in response to the "
                "independent critic. Address every concern and revision "
                "instruction that is supported by the supplied facts.\n\n"
                "<prior_decision>\n"
                f"{prior_decision.model_dump_json(indent=2)}\n"
                "</prior_decision>\n\n"
                "<critic_feedback>\n"
                f"{critique.model_dump_json(indent=2)}\n"
                "</critic_feedback>"
            )

        return (
            "Review the following accounts-payable case.\n\n"
            "<invoice>\n"
            f"{invoice.model_dump_json(indent=2)}\n"
            "</invoice>\n\n"
            "<validation_result>\n"
            f"{validation_result.model_dump_json(indent=2)}\n"
            "</validation_result>\n\n"
            "<approval_policy>\n"
            f"{policy.model_dump_json(indent=2)}\n"
            "</approval_policy>\n\n"
            "<decision_context>\n"
            f"{revision_context}\n"
            "</decision_context>"
        )

    @staticmethod
    def _coerce_decision(
        response: Any,
    ) -> ApprovalDecision:
        if isinstance(response, ApprovalDecision):
            return response

        try:
            return ApprovalDecision.model_validate(response)
        except ValidationError as exc:
            raise ApprovalDecisionError(
                (
                    "Model returned an invalid approval "
                    f"decision: {exc}"
                )
            ) from exc


class ApprovalCritic:
    """Independently review an approval decision."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self._structured_model = model.with_structured_output(
            ApprovalCritique
        )
        self._max_attempts = max_attempts

    def review(
        self,
        invoice: Invoice,
        validation_result: ValidationResult,
        policy: ApprovalPolicyAssessment,
        decision: ApprovalDecision,
    ) -> ApprovalCritique:
        """Review the approval decision and accept it or request revision."""

        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Audit the proposed approval decision.\n\n"
                    "<invoice>\n"
                    f"{invoice.model_dump_json(indent=2)}\n"
                    "</invoice>\n\n"
                    "<validation_result>\n"
                    f"{validation_result.model_dump_json(indent=2)}\n"
                    "</validation_result>\n\n"
                    "<approval_policy>\n"
                    f"{policy.model_dump_json(indent=2)}\n"
                    "</approval_policy>\n\n"
                    "<proposed_decision>\n"
                    f"{decision.model_dump_json(indent=2)}\n"
                    "</proposed_decision>"
                )
            ),
        ]

        errors: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._structured_model.invoke(
                    messages
                )
                critique = self._coerce_critique(response)

                critique_violations = self._critique_violations(
                    critique=critique,
                    decision=decision,
                    policy=policy,
                )

                if critique_violations:
                    raise ApprovalCritiqueError(
                        "; ".join(critique_violations)
                    )

                return critique

            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_message)

                if attempt == self._max_attempts:
                    break

                messages.append(
                    HumanMessage(
                        content=(
                            "The previous critique did not satisfy "
                            "the required schema or critic controls.\n\n"
                            f"Problem: {error_message}\n\n"
                            "Return a corrected independent critique."
                        )
                    )
                )

        raise ApprovalCritiqueError(
            (
                "Approval critique failed after "
                f"{self._max_attempts} attempts: "
                f"{' | '.join(errors)}"
            )
        )

    @staticmethod
    def _critique_violations(
        *,
        critique: ApprovalCritique,
        decision: ApprovalDecision,
        policy: ApprovalPolicyAssessment,
    ) -> list[str]:
        violations: list[str] = []

        if (
            critique.verdict == "revise"
            and not critique.revision_instructions
        ):
            violations.append(
                "a revise verdict must include revision instructions"
            )

        if (
            critique.verdict == "accept"
            and critique.revision_instructions
        ):
            violations.append(
                "an accept verdict must not include revision instructions"
            )

        decision_violations = (
            approval_decision_policy_violations(
                decision,
                policy,
            )
        )

        if (
            decision_violations
            and critique.verdict != "revise"
        ):
            violations.append(
                (
                    "the critic cannot accept a decision that violates "
                    "hard policy controls: "
                    f"{'; '.join(decision_violations)}"
                )
            )

        return violations

    @staticmethod
    def _coerce_critique(
        response: Any,
    ) -> ApprovalCritique:
        if isinstance(response, ApprovalCritique):
            return response

        try:
            return ApprovalCritique.model_validate(response)
        except ValidationError as exc:
            raise ApprovalCritiqueError(
                (
                    "Model returned an invalid approval "
                    f"critique: {exc}"
                )
            ) from exc
