from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from invoice_system.documents import load_document
from invoice_system.validation import validate_invoice
from invoice_system.workflow_nodes import (
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


def route_after_stage(
    state: WorkflowState,
) -> RouteDecision:
    """
    Stop after a technical failure.

    Business validation failures do not set status='failed', so they are not
    treated as workflow execution failures.
    """

    if state.get("status") == "failed":
        return "stop"

    return "continue"


def build_invoice_workflow(
    ingestion_agent: InvoiceExtractor,
    *,
    database_path: str | Path,
    document_loader: DocumentLoader = load_document,
    invoice_validator: InvoiceValidator = validate_invoice,
) -> CompiledStateGraph:
    """
    Build and compile the invoice processing workflow.
    """

    dependencies = WorkflowDependencies(
        ingestion_agent=ingestion_agent,
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

    builder.add_edge(
        "validate_invoice",
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
        "audit_events": [],
        "errors": [],
    }

    result = workflow.invoke(initial_state)

    return cast(WorkflowState, result)