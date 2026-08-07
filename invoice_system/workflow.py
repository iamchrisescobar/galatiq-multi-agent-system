from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from invoice_system.documents import load_document
from invoice_system.outcome_records import (
    DEFAULT_MANUAL_REVIEW_LOG_PATH,
    DEFAULT_REJECTION_LOG_PATH,
    log_manual_review,
    log_rejection,
)
from invoice_system.payment import mock_payment
from invoice_system.validation import validate_invoice
from invoice_system.workflow_nodes import (
    ApprovalDecider,
    ApprovalReviewer,
    DocumentLoader,
    InvoiceExtractor,
    InvoiceValidator,
    ManualReviewLogger,
    PaymentProcessor,
    RejectionLogger,
    WorkflowDependencies,
    WorkflowNodes,
)
from invoice_system.workflow_state import WorkflowState


RouteDecision = Literal[
    "continue",
    "stop",
]

ApprovalRoute = Literal[
    "accept",
    "revise",
    "manual_review",
    "stop",
]

FinalDecisionRoute = Literal[
    "payment",
    "rejection",
    "stop",
]


def route_after_stage(
    state: WorkflowState,
) -> RouteDecision:
    """Stop the graph after a technical workflow failure."""

    if state.get("status") == "failed":
        return "stop"

    return "continue"


def route_after_approval_critique(
    state: WorkflowState,
    *,
    max_revisions: int,
) -> ApprovalRoute:
    """Route the bounded approval <-> critic reflection loop."""

    if state.get("status") == "failed":
        return "stop"

    critique = state.get("approval_critique")

    if critique is None:
        raise RuntimeError(
            "Approval critique is missing from workflow state."
        )

    if critique.verdict == "accept":
        return "accept"

    revision_count = state.get(
        "approval_revision_count",
        0,
    )

    if revision_count >= max_revisions:
        return "manual_review"

    return "revise"


def route_after_approval_finalization(
    state: WorkflowState,
) -> FinalDecisionRoute:
    """Route a critic-accepted final decision to its terminal handler."""

    if state.get("status") == "failed":
        return "stop"

    decision = state.get("approval_decision")

    if decision is None:
        raise RuntimeError(
            "Approval decision is missing from workflow state."
        )

    if decision.decision == "approve":
        return "payment"

    if decision.decision == "reject":
        return "rejection"

    raise RuntimeError(
        f"Unsupported final approval decision: {decision.decision}."
    )


def build_invoice_workflow(
    ingestion_agent: InvoiceExtractor,
    approval_agent: ApprovalDecider,
    approval_critic: ApprovalReviewer,
    *,
    database_path: str | Path,
    max_approval_revisions: int = 2,
    payment_processor: PaymentProcessor = mock_payment,
    rejection_logger: RejectionLogger = log_rejection,
    manual_review_logger: ManualReviewLogger = log_manual_review,
    rejection_log_path: str | Path = DEFAULT_REJECTION_LOG_PATH,
    manual_review_log_path: str | Path = DEFAULT_MANUAL_REVIEW_LOG_PATH,
    document_loader: DocumentLoader = load_document,
    invoice_validator: InvoiceValidator = validate_invoice,
) -> CompiledStateGraph:
    """Build and compile the invoice processing workflow."""

    if max_approval_revisions < 0:
        raise ValueError(
            "max_approval_revisions cannot be negative."
        )

    dependencies = WorkflowDependencies(
        ingestion_agent=ingestion_agent,
        approval_agent=approval_agent,
        approval_critic=approval_critic,
        database_path=database_path,
        payment_processor=payment_processor,
        rejection_logger=rejection_logger,
        manual_review_logger=manual_review_logger,
        rejection_log_path=rejection_log_path,
        manual_review_log_path=manual_review_log_path,
        document_loader=document_loader,
        invoice_validator=invoice_validator,
    )

    nodes = WorkflowNodes(dependencies)

    builder = StateGraph(WorkflowState)

    builder.add_node(
        "load_document",
        nodes.load_document,
    )
    builder.add_node(
        "ingest_invoice",
        nodes.ingest_invoice,
    )
    builder.add_node(
        "validate_invoice",
        nodes.validate_invoice,
    )
    builder.add_node(
        "assess_approval",
        nodes.assess_approval,
    )
    builder.add_node(
        "make_approval_decision",
        nodes.make_approval_decision,
    )
    builder.add_node(
        "review_approval_decision",
        nodes.review_approval_decision,
    )
    builder.add_node(
        "finalize_approval",
        nodes.finalize_approval,
    )
    builder.add_node(
        "process_payment",
        nodes.process_payment,
    )
    builder.add_node(
        "handle_rejection",
        nodes.handle_rejection,
    )
    builder.add_node(
        "manual_review",
        nodes.escalate_manual_review,
    )

    builder.add_edge(
        START,
        "load_document",
    )

    builder.add_conditional_edges(
        "load_document",
        route_after_stage,
        {
            "continue": "ingest_invoice",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "ingest_invoice",
        route_after_stage,
        {
            "continue": "validate_invoice",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "validate_invoice",
        route_after_stage,
        {
            "continue": "assess_approval",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "assess_approval",
        route_after_stage,
        {
            "continue": "make_approval_decision",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "make_approval_decision",
        route_after_stage,
        {
            "continue": "review_approval_decision",
            "stop": END,
        },
    )

    def approval_critique_router(
        state: WorkflowState,
    ) -> ApprovalRoute:
        return route_after_approval_critique(
            state,
            max_revisions=max_approval_revisions,
        )

    builder.add_conditional_edges(
        "review_approval_decision",
        approval_critique_router,
        {
            "accept": "finalize_approval",
            "revise": "make_approval_decision",
            "manual_review": "manual_review",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "finalize_approval",
        route_after_approval_finalization,
        {
            "payment": "process_payment",
            "rejection": "handle_rejection",
            "stop": END,
        },
    )

    builder.add_edge(
        "process_payment",
        END,
    )

    builder.add_edge(
        "handle_rejection",
        END,
    )

    builder.add_edge(
        "manual_review",
        END,
    )

    return builder.compile()


def run_invoice_workflow(
    workflow: CompiledStateGraph,
    invoice_path: str | Path,
) -> WorkflowState:
    """Run one invoice through the compiled workflow."""

    initial_state: WorkflowState = {
        "invoice_path": str(invoice_path),
        "status": "running",
        "approval_revision_count": 0,
        "audit_events": [],
        "errors": [],
    }

    result = workflow.invoke(initial_state)

    return cast(WorkflowState, result)
