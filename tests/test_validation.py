from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_system.database import initialize_database
from invoice_system.models import Invoice, InvoiceItem
from invoice_system.validation import validate_invoice


@pytest.fixture
def inventory_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "inventory.db"
    initialize_database(database_path)
    return database_path


def make_invoice(
    *,
    invoice_number: str = "INV-TEST",
    vendor: str | None = "Test Vendor",
    amount: Decimal | None = Decimal("1000.00"),
    items: list[InvoiceItem] | None = None,
    invoice_date: date | None = date(2026, 8, 1),
    due_date: date | None = date(2026, 9, 1),
) -> Invoice:
    return Invoice(
        invoice_number=invoice_number,
        vendor=vendor,
        amount=amount,
        items=items if items is not None else [],
        invoice_date=invoice_date,
        due_date=due_date,
    )


@pytest.mark.parametrize(
    ("invoice_number", "items"),
    [
        (
            "INV-1001",
            [InvoiceItem(name="WidgetA", quantity=10)],
        ),
        (
            "INV-1004",
            [InvoiceItem(name="WidgetB", quantity=5)],
        ),
        (
            "INV-1006",
            [
                InvoiceItem(name="WidgetA", quantity=5),
                InvoiceItem(name="WidgetB", quantity=3),
            ],
        ),
    ],
)
def test_valid_inventory_scenarios_pass(
    inventory_database: Path,
    invoice_number: str,
    items: list[InvoiceItem],
) -> None:
    invoice = make_invoice(
        invoice_number=invoice_number,
        items=items,
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is True
    assert result.issues == []


def test_inv_1002_fails_when_quantity_exceeds_stock(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        invoice_number="INV-1002",
        items=[InvoiceItem(name="GadgetX", quantity=20)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert len(result.issues) == 1

    issue = result.issues[0]
    assert issue.code == "insufficient_stock"
    assert issue.item == "GadgetX"
    assert issue.requested_quantity == 20
    assert issue.available_stock == 5


def test_inv_1003_fails_for_zero_stock_item(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        invoice_number="INV-1003",
        items=[InvoiceItem(name="FakeItem", quantity=1)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert result.issues[0].code == "out_of_stock"
    assert result.issues[0].available_stock == 0


@pytest.mark.parametrize(
    ("invoice_number", "unknown_item"),
    [
        ("INV-1008", "SuperGizmo"),
        ("INV-1008", "MegaSprocket"),
        ("INV-1016", "WidgetC"),
    ],
)
def test_unknown_items_are_rejected(
    inventory_database: Path,
    invoice_number: str,
    unknown_item: str,
) -> None:
    invoice = make_invoice(
        invoice_number=invoice_number,
        items=[InvoiceItem(name=unknown_item, quantity=1)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert result.issues[0].code == "unknown_item"
    assert result.issues[0].item == unknown_item


def test_inv_1009_fails_for_negative_quantity(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        invoice_number="INV-1009",
        items=[InvoiceItem(name="WidgetA", quantity=-4)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert result.issues[0].code == "invalid_quantity"
    assert result.issues[0].requested_quantity == -4


def test_missing_required_invoice_data_is_reported(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        vendor=None,
        amount=None,
        invoice_date=None,
        due_date=None,
        items=[],
    )

    result = validate_invoice(invoice, inventory_database)

    issue_codes = {issue.code for issue in result.issues}

    assert result.passed is False
    assert issue_codes == {
        "missing_vendor",
        "missing_amount",
        "missing_invoice_date",
        "missing_due_date",
        "missing_items",
    }


def test_inventory_lookup_is_case_insensitive(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        items=[InvoiceItem(name="  widgeta  ", quantity=15)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is True

def test_missing_invoice_date_is_rejected(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        invoice_date=None,
        items=[InvoiceItem(name="WidgetA", quantity=1)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert len(result.issues) == 1
    assert result.issues[0].code == "missing_invoice_date"
    assert (
        result.issues[0].message
        == "The invoice does not contain a valid invoice date."
    )
    
def test_due_date_before_invoice_date_is_rejected(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        invoice_date=date(2026, 2, 15),
        due_date=date(2026, 2, 1),
        items=[InvoiceItem(name="WidgetA", quantity=1)],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert len(result.issues) == 1

    issue = result.issues[0]
    assert issue.code == "due_date_before_invoice_date"
    assert (
        issue.message
        == "Due date 2026-02-01 cannot be earlier than invoice date 2026-02-15."
    )

def test_repeated_line_items_are_aggregated_before_stock_validation(
    inventory_database: Path,
) -> None:
    """INV-1013-style repeated lines must consume inventory cumulatively."""

    invoice = make_invoice(
        invoice_number="INV-1013",
        amount=Decimal("22562.80"),
        items=[
            InvoiceItem(name="WidgetA", quantity=15),
            InvoiceItem(name="WidgetB", quantity=10),
            InvoiceItem(name="GadgetX", quantity=5),
            InvoiceItem(name="WidgetA", quantity=5),
            InvoiceItem(name="WidgetB", quantity=8),
            InvoiceItem(name="GadgetX", quantity=3),
            InvoiceItem(name="WidgetA", quantity=2),
            InvoiceItem(name="GadgetX", quantity=1),
        ],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert len(result.issues) == 3

    issues_by_item = {
        issue.item: issue
        for issue in result.issues
    }

    widget_a_issue = issues_by_item["WidgetA"]
    assert widget_a_issue.code == "insufficient_stock"
    assert widget_a_issue.requested_quantity == 22
    assert widget_a_issue.available_stock == 15

    widget_b_issue = issues_by_item["WidgetB"]
    assert widget_b_issue.code == "insufficient_stock"
    assert widget_b_issue.requested_quantity == 18
    assert widget_b_issue.available_stock == 10

    gadget_x_issue = issues_by_item["GadgetX"]
    assert gadget_x_issue.code == "insufficient_stock"
    assert gadget_x_issue.requested_quantity == 9
    assert gadget_x_issue.available_stock == 5


def test_repeated_line_items_pass_when_combined_quantity_is_within_stock(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        items=[
            InvoiceItem(name="WidgetA", quantity=8),
            InvoiceItem(name="WidgetA", quantity=7),
        ],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is True
    assert result.issues == []


def test_repeated_item_case_variants_are_aggregated(
    inventory_database: Path,
) -> None:
    invoice = make_invoice(
        items=[
            InvoiceItem(name="WidgetA", quantity=10),
            InvoiceItem(name="  widgeta  ", quantity=6),
        ],
    )

    result = validate_invoice(invoice, inventory_database)

    assert result.passed is False
    assert len(result.issues) == 1

    issue = result.issues[0]
    assert issue.code == "insufficient_stock"
    assert issue.item == "WidgetA"
    assert issue.requested_quantity == 16
    assert issue.available_stock == 15
