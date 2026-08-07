from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from invoice_system.agents.approval import (
    ApprovalAgent,
    ApprovalCritic,
)
from invoice_system.approval import (
    ApprovalCritique,
    ApprovalDecision,
    assess_approval_policy,
)
from invoice_system.config import LLMSettings
from invoice_system.database import initialize_database
from invoice_system.llm import create_chat_model
from invoice_system.models import Invoice, InvoiceItem
from invoice_system.validation import validate_invoice


DATABASE_PATH = PROJECT_ROOT / "inventory.db"
MAX_REVISIONS = 2


def main() -> None:
    settings = LLMSettings.from_env()
    initialize_database(DATABASE_PATH)

    invoice = Invoice(
        invoice_number="INV-HIGH-VALUE-SMOKE",
        vendor="Atlas Industrial Supply",
        amount=Decimal("12500.00"),
        items=[
            InvoiceItem(name="WidgetA", quantity=10),
            InvoiceItem(name="WidgetB", quantity=5),
            InvoiceItem(name="GadgetX", quantity=3),
        ],
        invoice_date=date(2026, 1, 24),
        due_date=date(2026, 2, 24),
    )

    validation_result = validate_invoice(
        invoice,
        DATABASE_PATH,
    )

    policy = assess_approval_policy(
        invoice,
        validation_result,
    )

    model = create_chat_model(settings)

    approval_agent = ApprovalAgent(
        model,
        max_attempts=2,
    )

    approval_critic = ApprovalCritic(
        model,
        max_attempts=2,
    )

    prior_decision: ApprovalDecision | None = None
    prior_critique: ApprovalCritique | None = None
    revision_count = 0

    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")

    print("\nPolicy assessment")
    print(policy.model_dump_json(indent=2))

    while True:
        decision = approval_agent.decide(
            invoice,
            validation_result,
            policy,
            prior_decision=prior_decision,
            critique=prior_critique,
        )

        print(
            f"\nApproval decision "
            f"#{revision_count + 1}"
        )
        print(decision.model_dump_json(indent=2))

        critique = approval_critic.review(
            invoice,
            validation_result,
            policy,
            decision,
        )

        print(
            f"\nIndependent critique "
            f"#{revision_count + 1}"
        )
        print(critique.model_dump_json(indent=2))

        if critique.verdict == "accept":
            print(
                "\nFinal result: critic accepted the "
                f"{decision.decision} decision."
            )
            break

        if revision_count >= MAX_REVISIONS:
            print(
                "\nFinal result: manual review required; "
                "the bounded reflection loop did not converge."
            )
            break

        prior_decision = decision
        prior_critique = critique
        revision_count += 1


if __name__ == "__main__":
    main()
