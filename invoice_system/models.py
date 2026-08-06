from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InvoiceItem(BaseModel):
    """
    An item extracted from an invoice.

    Fields are optional because invoices may contain missing or
    malformed information. Business validation is later responsible for 
    determining whether the item is acceptable.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = None
    quantity: int | None = None


class Invoice(BaseModel):
    """
    Schema of an invoice.

    Optional fields allow the ingestion stage to preserve incomplete invoices
    rather than hallucinating values or failing before validation can explain the
    problem.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    invoice_number: str | None = None
    vendor: str | None = None
    amount: Decimal | None = None
    items: list[InvoiceItem] = Field(default_factory=list)
    invoice_date: date | None = None
    due_date: date | None = None


ValidationIssueCode = Literal[
    "missing_vendor",
    "missing_amount",
    "invalid_amount",
    "missing_invoice_date",
    "missing_due_date",
    "due_date_before_invoice_date",
    "missing_items",
    "missing_item_name",
    "invalid_quantity",
    "unknown_item",
    "out_of_stock",
    "insufficient_stock",
]


class ValidationIssue(BaseModel):
    code: ValidationIssueCode
    message: str
    item: str | None = None
    requested_quantity: int | None = None
    available_stock: int | None = None


class ValidationResult(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)