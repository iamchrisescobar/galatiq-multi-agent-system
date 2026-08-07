from __future__ import annotations

from decimal import Decimal

import pytest

from invoice_system.payment import mock_payment


def test_mock_payment_returns_success() -> None:
    result = mock_payment(
        "Widgets Inc.",
        Decimal("5000.00"),
    )

    assert result.status == "success"
    assert result.vendor == "Widgets Inc."
    assert result.amount == Decimal("5000.00")


def test_mock_payment_rejects_empty_vendor() -> None:
    with pytest.raises(
        ValueError,
        match="vendor must not be empty",
    ):
        mock_payment(
            "   ",
            Decimal("5000.00"),
        )


def test_mock_payment_rejects_non_positive_amount() -> None:
    with pytest.raises(
        ValueError,
        match="amount must be greater than zero",
    ):
        mock_payment(
            "Widgets Inc.",
            Decimal("0.00"),
        )
