from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from invoice_system.documents import load_document
from invoice_system.validation import validate_invoice
from invoice_system.workflow_nodes import (
    ApprovalDecider,
    ApprovalReviewer,
    DocumentLoader,
    InvoiceExtractor,
    InvoiceValidator,
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


def build_invoice_workflow(
    ingestion_agent: InvoiceExtractor,
    approval_agent: ApprovalDecider,
    approval_critic: ApprovalReviewer,
    *,
    database_path: str | Path,
    max_approval_revisions: int = 2,
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

    builder.add_edge(
        "finalize_approval",
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
