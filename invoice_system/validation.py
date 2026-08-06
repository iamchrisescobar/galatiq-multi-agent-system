from __future__ import annotations

from pathlib import Path

from invoice_system.database import DEFAULT_DATABASE_PATH, lookup_inventory
from invoice_system.models import Invoice, ValidationIssue, ValidationResult


def validate_invoice(
    invoice: Invoice,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> ValidationResult:
    """
    Validate an extracted invoice against business rules and inventory.

    This function contains no LLM reasoning. Every approval workflow will rely
    on these deterministic checks as its source of truth.
    """

    issues: list[ValidationIssue] = []

    _validate_invoice_fields(invoice, issues)
    _validate_items(invoice, database_path, issues)

    return ValidationResult(
        passed=len(issues) == 0,
        issues=issues,
    )


def _validate_invoice_fields(
    invoice: Invoice,
    issues: list[ValidationIssue],
) -> None:
    if not invoice.vendor or not invoice.vendor.strip():
        issues.append(
            ValidationIssue(
                code="missing_vendor",
                message="The invoice does not contain a vendor.",
            )
        )

    if invoice.amount is None:
        issues.append(
            ValidationIssue(
                code="missing_amount",
                message="The invoice does not contain an amount.",
            )
        )
    elif invoice.amount <= 0:
        issues.append(
            ValidationIssue(
                code="invalid_amount",
                message=f"Invoice amount must be positive and non-zero; received {invoice.amount}.",
            )
        )
        
    if invoice.invoice_date is None:
        issues.append(
            ValidationIssue(
                code="missing_invoice_date",
                message="The invoice does not contain a valid invoice date.",
            )
        )

    if invoice.due_date is None:
        issues.append(
            ValidationIssue(
                code="missing_due_date",
                message="The invoice does not contain a valid due date.",
            )
        )

    if (
        invoice.invoice_date is not None
        and invoice.due_date is not None
        and invoice.due_date < invoice.invoice_date
    ):
        issues.append(
            ValidationIssue(
                code="due_date_before_invoice_date",
                message=(
                    f"Due date {invoice.due_date.isoformat()} cannot be earlier "
                    f"than invoice date {invoice.invoice_date.isoformat()}."
                ),
            )
        )

    if not invoice.items:
        issues.append(
            ValidationIssue(
                code="missing_items",
                message="The invoice does not contain any line items.",
            )
        )


def _validate_items(
    invoice: Invoice,
    database_path: str | Path,
    issues: list[ValidationIssue],
) -> None:
    for position, item in enumerate(invoice.items, start=1):
        item_label = item.name or f"line item {position}"

        if not item.name or not item.name.strip():
            issues.append(
                ValidationIssue(
                    code="missing_item_name",
                    item=item.name,
                    requested_quantity=item.quantity,
                    message=f"Line item {position} does not contain an item name.",
                )
            )
            continue

        if item.quantity is None or item.quantity <= 0:
            issues.append(
                ValidationIssue(
                    code="invalid_quantity",
                    item=item.name,
                    requested_quantity=item.quantity,
                    message=(
                        f"{item.name} has an invalid quantity: "
                        f"{item.quantity!r}. Quantity must be a positive integer."
                    ),
                )
            )
            continue

        inventory_record = lookup_inventory(
            item_name=item.name,
            database_path=database_path,
        )

        if inventory_record is None:
            issues.append(
                ValidationIssue(
                    code="unknown_item",
                    item=item.name,
                    requested_quantity=item.quantity,
                    message=f"{item.name} was not found in inventory.",
                )
            )
            continue

        available_stock = int(inventory_record["stock"])
        canonical_item_name = str(inventory_record["item"])

        if available_stock <= 0:
            issues.append(
                ValidationIssue(
                    code="out_of_stock",
                    item=canonical_item_name,
                    requested_quantity=item.quantity,
                    available_stock=available_stock,
                    message=f"{canonical_item_name} is out of stock.",
                )
            )
            continue

        if item.quantity > available_stock:
            issues.append(
                ValidationIssue(
                    code="insufficient_stock",
                    item=canonical_item_name,
                    requested_quantity=item.quantity,
                    available_stock=available_stock,
                    message=(
                        f"{canonical_item_name} requested quantity "
                        f"{item.quantity} exceeds available stock "
                        f"{available_stock}."
                    ),
                )
            )