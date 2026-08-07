from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from invoice_system.agents.approval import (
    ApprovalAgent,
    ApprovalCritic,
)
from invoice_system.agents.ingestion import IngestionAgent
from invoice_system.config import LLMSettings
from invoice_system.database import initialize_database
from invoice_system.llm import create_chat_model
from invoice_system.workflow import (
    build_invoice_workflow,
    run_invoice_workflow,
)


SAMPLE_INVOICE = (
    PROJECT_ROOT
    / "data"
    / "invoices"
    / "invoice_1005.json"
)

DATABASE_PATH = PROJECT_ROOT / "inventory.db"


def main() -> None:
    settings = LLMSettings.from_env()
    initialize_database(DATABASE_PATH)

    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"Invoice: {SAMPLE_INVOICE.name}")

    model = create_chat_model(settings)

    ingestion_agent = IngestionAgent(
        model,
        max_attempts=2,
    )
    approval_agent = ApprovalAgent(
        model,
        max_attempts=2,
    )
    approval_critic = ApprovalCritic(
        model,
        max_attempts=2,
    )

    workflow = build_invoice_workflow(
        ingestion_agent,
        approval_agent,
        approval_critic,
        database_path=DATABASE_PATH,
        max_approval_revisions=2,
    )

    result = run_invoice_workflow(
        workflow,
        SAMPLE_INVOICE,
    )

    print("\nWorkflow result")
    print(f"Status: {result['status']}")
    print(f"Current stage: {result['current_stage']}")

    invoice = result.get("invoice")

    if invoice is not None:
        print("\nExtracted invoice")
        print(invoice.model_dump_json(indent=2))

    validation_result = result.get(
        "validation_result"
    )

    if validation_result is not None:
        print("\nValidation result")
        print(validation_result.model_dump_json(indent=2))

    approval_policy = result.get("approval_policy")

    if approval_policy is not None:
        print("\nApproval policy")
        print(approval_policy.model_dump_json(indent=2))

    approval_decision = result.get("approval_decision")

    if approval_decision is not None:
        print("\nApproval decision")
        print(approval_decision.model_dump_json(indent=2))

    approval_critique = result.get("approval_critique")

    if approval_critique is not None:
        print("\nLatest approval critique")
        print(approval_critique.model_dump_json(indent=2))

    print(
        "\nApproval revisions: "
        f"{result.get('approval_revision_count', 0)}"
    )

    print("\nAudit events")

    for event in result.get("audit_events", []):
        print(
            f"- {event.stage}: "
            f"{event.status} "
            f"({event.duration_ms} ms) - "
            f"{event.message}"
        )

    errors = result.get("errors", [])

    if errors:
        print("\nErrors")

        for error in errors:
            print(
                f"- {error.stage}: "
                f"{error.error_type}: "
                f"{error.message}"
            )


if __name__ == "__main__":
    main()
