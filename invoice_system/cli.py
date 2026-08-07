from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


SEPARATOR = "-" * 50


def _value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _format_currency(value: Any) -> str:
    if value is None:
        return "Unknown"

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)

    return f"${amount:,.2f}"


def _format_quantity(value: Any) -> str:
    if value is None:
        return "unknown"

    return str(value)


def _format_validation_issue(issue: Any) -> str:
    code = _value(issue, "code")

    if code == "insufficient_stock":
        item = _value(issue, "item")
        requested = _value(issue, "requested_quantity")
        available = _value(issue, "available_stock")

        if (
            item is not None
            and requested is not None
            and available is not None
        ):
            return (
                f"{item}: requested "
                f"{_format_quantity(requested)}, available "
                f"{_format_quantity(available)}"
            )

    message = _value(issue, "message")

    if message:
        return str(message)

    if code:
        return str(code)

    return str(issue)


def _risk_level(result: Mapping[str, Any]) -> str | None:
    approval_decision = result.get("approval_decision")
    risk_level = _value(approval_decision, "risk_level")

    if risk_level is None:
        return None

    return str(risk_level).upper()


def _outcome(result: Mapping[str, Any]) -> str:
    status = str(result.get("status", "")).lower()

    if status == "manual_review":
        return "MANUAL REVIEW"

    if status == "failed":
        return "FAILED"

    payment_result = result.get("payment_result")
    payment_status = str(
        _value(payment_result, "status", "")
    ).lower()

    if payment_status == "success":
        return "PAID"

    if result.get("rejection_record") is not None:
        return "REJECTED"

    approval_decision = result.get("approval_decision")
    decision = str(
        _value(approval_decision, "decision", "")
    ).lower()

    if status == "completed" and decision == "reject":
        return "REJECTED"

    if status == "completed" and decision == "approve":
        return "APPROVED"

    if status:
        return status.upper()

    return "UNKNOWN"


def _manual_review_reason(
    result: Mapping[str, Any],
) -> str:
    record = result.get("manual_review_record")
    reason = _value(record, "reason")

    if reason:
        return str(reason)

    revision_count = result.get(
        "approval_revision_count",
        0,
    )

    return (
        "Automated approval could not converge after "
        f"{revision_count} revision(s)."
    )


def _format_error(error: Any) -> str:
    stage = _value(error, "stage")
    error_type = _value(error, "error_type")
    message = _value(error, "message")

    prefix_parts = [
        str(part)
        for part in (stage, error_type)
        if part
    ]

    if prefix_parts and message:
        return f"{' / '.join(prefix_parts)}: {message}"

    if message:
        return str(message)

    return str(error)


def format_business_summary(
    result: Mapping[str, Any],
    *,
    rejection_log_path: Path | str = (
        Path("data") / "rejections.jsonl"
    ),
    manual_review_log_path: Path | str = (
        Path("data") / "manual_reviews.jsonl"
    ),
    elapsed_seconds: float | None = None,
) -> str:
    lines: list[str] = [
        "Invoice Processing Result",
        SEPARATOR,
        "",
    ]

    invoice = result.get("invoice")

    if invoice is not None:
        invoice_number = _value(
            invoice,
            "invoice_number",
            "Unknown",
        )
        vendor = _value(invoice, "vendor", "Unknown")
        amount = _value(invoice, "amount")

        lines.extend(
            [
                f"Invoice:   {invoice_number}",
                f"Vendor:    {vendor}",
                f"Amount:    {_format_currency(amount)}",
                "",
            ]
        )

    outcome = _outcome(result)
    lines.append(f"Outcome:   {outcome}")

    risk_level = _risk_level(result)

    if risk_level is not None:
        lines.append(f"Risk:      {risk_level}")

    validation_result = result.get("validation_result")
    issues = list(
        _value(
            validation_result,
            "issues",
            [],
        )
        or []
    )

    if validation_result is not None:
        lines.append("")

        if issues:
            lines.append("Validation issues:")

            for issue in issues:
                lines.append(
                    f"  - {_format_validation_issue(issue)}"
                )
        else:
            passed = bool(
                _value(
                    validation_result,
                    "passed",
                    False,
                )
            )
            lines.append(
                "Validation: Passed"
                if passed
                else "Validation: Failed"
            )

    approval_decision = result.get(
        "approval_decision"
    )

    if outcome == "PAID":
        lines.extend(
            [
                "",
                "Decision:   Approved",
            ]
        )

        payment_result = result.get(
            "payment_result"
        )
        payment_amount = _value(
            payment_result,
            "amount",
            _value(invoice, "amount"),
        )
        payment_vendor = _value(
            payment_result,
            "vendor",
            _value(invoice, "vendor", "Unknown"),
        )

        lines.extend(
            [
                "",
                "Payment:",
                (
                    f"  Paid {_format_currency(payment_amount)} "
                    f"to {payment_vendor}"
                ),
            ]
        )

    elif outcome == "REJECTED":
        decision_summary = _value(
            approval_decision,
            "summary",
            "Invoice rejected.",
        )

        lines.extend(
            [
                "",
                "Decision:",
                f"  {decision_summary}",
                "",
                "Payment:   Not processed",
                f"Record:    {rejection_log_path}",
            ]
        )

    elif outcome == "MANUAL REVIEW":
        lines.extend(
            [
                "",
                "Decision:   Pending human review",
                "",
                "Reason:",
                f"  {_manual_review_reason(result)}",
                "",
                "Payment:   Not processed",
                f"Queue:     {manual_review_log_path}",
            ]
        )

    elif outcome == "FAILED":
        current_stage = result.get("current_stage")

        if current_stage:
            lines.extend(
                [
                    "",
                    f"Stage:     {current_stage}",
                ]
            )

        errors = list(
            result.get("errors", [])
            or []
        )

        if errors:
            lines.extend(
                [
                    "",
                    "Errors:",
                ]
            )

            for error in errors:
                lines.append(
                    f"  - {_format_error(error)}"
                )

    else:
        decision = _value(
            approval_decision,
            "decision",
        )

        if decision:
            lines.extend(
                [
                    "",
                    f"Decision:   {str(decision).title()}",
                ]
            )

    if elapsed_seconds is not None:
        lines.extend(
            [
                "",
                (
                    "Processing time: "
                    f"{elapsed_seconds:.2f}s"
                ),
            ]
        )

    return "\n".join(lines)


def _to_json(value: Any) -> str:
    if value is None:
        return "null"

    model_dump_json = getattr(
        value,
        "model_dump_json",
        None,
    )

    if callable(model_dump_json):
        return model_dump_json(indent=2)

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):
        return json.dumps(
            model_dump(mode="json"),
            indent=2,
            default=str,
        )

    return json.dumps(
        value,
        indent=2,
        default=str,
    )


def format_verbose_details(
    result: Mapping[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
    invoice_path: Path | str | None = None,
    captured_stdout: str | None = None,
) -> str:
    lines: list[str] = [
        "Technical Details",
        SEPARATOR,
    ]

    if provider is not None:
        lines.append(f"Provider: {provider}")

    if model is not None:
        lines.append(f"Model: {model}")

    if invoice_path is not None:
        lines.append(f"Invoice path: {invoice_path}")

    sections = [
        ("Extracted invoice", result.get("invoice")),
        (
            "Validation result",
            result.get("validation_result"),
        ),
        (
            "Approval policy",
            result.get("approval_policy"),
        ),
        (
            "Approval decision",
            result.get("approval_decision"),
        ),
        (
            "Latest approval critique",
            result.get("approval_critique"),
        ),
        (
            "Payment result",
            result.get("payment_result"),
        ),
        (
            "Rejection record",
            result.get("rejection_record"),
        ),
        (
            "Manual review work item",
            result.get("manual_review_record"),
        ),
    ]

    for title, value in sections:
        if value is None:
            continue

        lines.extend(
            [
                "",
                title,
                _to_json(value),
            ]
        )

    lines.extend(
        [
            "",
            (
                "Approval revisions: "
                f"{result.get('approval_revision_count', 0)}"
            ),
            "",
            "Audit events",
        ]
    )

    audit_events = list(
        result.get("audit_events", [])
        or []
    )

    if audit_events:
        for event in audit_events:
            stage = _value(event, "stage", "unknown")
            status = _value(event, "status", "unknown")
            duration_ms = _value(
                event,
                "duration_ms",
                "unknown",
            )
            message = _value(event, "message", "")

            lines.append(
                f"- {stage}: {status} "
                f"({duration_ms} ms) - {message}"
            )
    else:
        lines.append("- none")

    errors = list(
        result.get("errors", [])
        or []
    )

    if errors:
        lines.extend(
            [
                "",
                "Errors",
            ]
        )

        for error in errors:
            lines.append(
                f"- {_format_error(error)}"
            )

    if captured_stdout:
        captured_stdout = captured_stdout.strip()

        if captured_stdout:
            lines.extend(
                [
                    "",
                    "Captured workflow stdout",
                    captured_stdout,
                ]
            )

    return "\n".join(lines)


def format_cli_output(
    result: Mapping[str, Any],
    *,
    verbose: bool = False,
    provider: str | None = None,
    model: str | None = None,
    invoice_path: Path | str | None = None,
    rejection_log_path: Path | str = (
        Path("data") / "rejections.jsonl"
    ),
    manual_review_log_path: Path | str = (
        Path("data") / "manual_reviews.jsonl"
    ),
    elapsed_seconds: float | None = None,
    captured_stdout: str | None = None,
) -> str:
    summary = format_business_summary(
        result,
        rejection_log_path=rejection_log_path,
        manual_review_log_path=manual_review_log_path,
        elapsed_seconds=elapsed_seconds,
    )

    if not verbose:
        return summary

    details = format_verbose_details(
        result,
        provider=provider,
        model=model,
        invoice_path=invoice_path,
        captured_stdout=captured_stdout,
    )

    return f"{summary}\n\n{details}"


def exit_code_for_result(
    result: Mapping[str, Any],
) -> int:
    status = str(
        result.get("status", "")
    ).lower()

    if status == "completed":
        return 0

    if status == "manual_review":
        return 2

    return 1
