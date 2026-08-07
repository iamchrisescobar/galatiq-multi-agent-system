from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PaymentStatus = Literal["success"]


class PaymentResult(BaseModel):
    """Result returned by the local mock payment processor."""

    model_config = ConfigDict(frozen=True)

    status: PaymentStatus
    vendor: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)


def mock_payment(
    vendor: str,
    amount: Decimal,
) -> PaymentResult:
    """Simulate the external banking payment side effect locally."""

    normalized_vendor = vendor.strip()

    if not normalized_vendor:
        raise ValueError("vendor must not be empty.")

    if amount <= 0:
        raise ValueError("amount must be greater than zero.")

    print(f"Paid {amount} to {normalized_vendor}")

    return PaymentResult(
        status="success",
        vendor=normalized_vendor,
        amount=amount,
    )
