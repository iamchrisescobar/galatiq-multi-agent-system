from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from invoice_system.database import DEFAULT_DATABASE_PATH, lookup_inventory
from invoice_system.models import Invoice, ValidationIssue, ValidationResult


_TRAILING_QUALIFIER_RE = re.compile(
    r"^(?P<base>.+?)\s*\((?P<qualifier>[^()]*)\)\s*$"
)

_OPERATIONAL_ITEM_QUALIFIERS = {
    "rush",
    "rush order",
    "expedited",
    "expedited order",
    "replacement",
    "sample",
}


@dataclass
class _AggregatedLineItem:
    """Combined quantity for one resolved product across invoice lines."""

    lookup_name: str
    requested_quantity: int
    available_stock: int | None


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
                message=(
                    "Invoice amount must be positive and non-zero; "
                    f"received {invoice.amount}."
                ),
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



def _lookup_inventory_identity(
    item_name: str,
    database_path: str | Path,
) -> tuple[str, int] | None:
    """
    Resolve exact inventory names first, then a whitespace-only variant.

    The fallback is intentionally narrow: it removes internal whitespace but
    does not alter punctuation, spelling, or arbitrary characters. This lets
    OCR/presentation variants such as "Widget A" resolve to "WidgetA" while
    avoiding fuzzy matching of genuinely unknown products.
    """

    exact_record = lookup_inventory(
        item_name=item_name,
        database_path=database_path,
    )

    if exact_record is not None:
        return (
            str(exact_record["item"]),
            int(exact_record["stock"]),
        )

    compact_name = re.sub(r"\s+", "", item_name)

    if compact_name.casefold() == item_name.casefold():
        return None

    compact_record = lookup_inventory(
        item_name=compact_name,
        database_path=database_path,
    )

    if compact_record is None:
        return None

    return (
        str(compact_record["item"]),
        int(compact_record["stock"]),
    )


def _resolve_inventory_identity(
    item_name: str,
    database_path: str | Path,
) -> tuple[str, int] | None:
    """
    Resolve a line-item description to a canonical inventory identity.

    Resolution is deliberately conservative:
    1. exact inventory match,
    2. whitespace-only normalization for OCR/presentation variants,
    3. recognized operational qualifiers in trailing parentheses.

    Product-like variants such as "WidgetA (XL)" are intentionally not
    collapsed into WidgetA because doing so could silently map a distinct SKU.
    """

    direct_match = _lookup_inventory_identity(
        item_name,
        database_path,
    )

    if direct_match is not None:
        return direct_match

    match = _TRAILING_QUALIFIER_RE.fullmatch(item_name)

    if match is None:
        return None

    qualifier = " ".join(
        match.group("qualifier").casefold().split()
    )

    if qualifier not in _OPERATIONAL_ITEM_QUALIFIERS:
        return None

    base_name = match.group("base").strip()

    if not base_name:
        return None

    return _lookup_inventory_identity(
        base_name,
        database_path,
    )


def _validate_items(
    invoice: Invoice,
    database_path: str | Path,
    issues: list[ValidationIssue],
) -> None:
    """
    Resolve inventory identity, then aggregate repeated product lines.

    Inventory is shared across the whole invoice, so repeated lines for the
    same canonical product must consume stock cumulatively. Operational
    qualifiers such as "WidgetA (rush order)" may resolve to WidgetA when the
    base item exists in inventory, allowing the quantities to be aggregated.

    Missing names and invalid quantities remain line-level data-integrity
    errors and are excluded from aggregation.
    """

    aggregated_items: dict[str, _AggregatedLineItem] = {}

    for position, item in enumerate(invoice.items, start=1):
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

        raw_name = item.name.strip()
        resolved = _resolve_inventory_identity(
            raw_name,
            database_path,
        )

        if resolved is None:
            lookup_name = raw_name
            available_stock = None
        else:
            lookup_name, available_stock = resolved

        normalized_key = lookup_name.casefold()
        existing_item = aggregated_items.get(normalized_key)

        if existing_item is None:
            aggregated_items[normalized_key] = _AggregatedLineItem(
                lookup_name=lookup_name,
                requested_quantity=item.quantity,
                available_stock=available_stock,
            )
        else:
            existing_item.requested_quantity += item.quantity

            # A canonical inventory resolution takes precedence if an earlier
            # unresolved spelling/description happened to share the same key.
            if (
                existing_item.available_stock is None
                and available_stock is not None
            ):
                existing_item.lookup_name = lookup_name
                existing_item.available_stock = available_stock

    for aggregated_item in aggregated_items.values():
        if aggregated_item.available_stock is None:
            issues.append(
                ValidationIssue(
                    code="unknown_item",
                    item=aggregated_item.lookup_name,
                    requested_quantity=aggregated_item.requested_quantity,
                    message=(
                        f"{aggregated_item.lookup_name} was not found in inventory."
                    ),
                )
            )
            continue

        available_stock = aggregated_item.available_stock
        canonical_item_name = aggregated_item.lookup_name

        if available_stock <= 0:
            issues.append(
                ValidationIssue(
                    code="out_of_stock",
                    item=canonical_item_name,
                    requested_quantity=aggregated_item.requested_quantity,
                    available_stock=available_stock,
                    message=f"{canonical_item_name} is out of stock.",
                )
            )
            continue

        if aggregated_item.requested_quantity > available_stock:
            issues.append(
                ValidationIssue(
                    code="insufficient_stock",
                    item=canonical_item_name,
                    requested_quantity=aggregated_item.requested_quantity,
                    available_stock=available_stock,
                    message=(
                        f"{canonical_item_name} requested quantity "
                        f"{aggregated_item.requested_quantity} exceeds available "
                        f"stock {available_stock}."
                    ),
                )
            )
