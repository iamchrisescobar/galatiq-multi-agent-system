from __future__ import annotations

import argparse
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from invoice_system.agents.approval import (
    ApprovalAgent,
    ApprovalCritic,
)
from invoice_system.agents.ingestion import IngestionAgent
from invoice_system.cli import (
    exit_code_for_result,
    format_cli_output,
)
from invoice_system.config import LLMSettings
from invoice_system.database import initialize_database
from invoice_system.llm import create_chat_model
from invoice_system.workflow import (
    build_invoice_workflow,
    run_invoice_workflow,
)


DATABASE_PATH = PROJECT_ROOT / "inventory.db"
REJECTION_LOG_PATH = PROJECT_ROOT / "data" / "rejections.jsonl"
MANUAL_REVIEW_LOG_PATH = (
    PROJECT_ROOT / "data" / "manual_reviews.jsonl"
)


class InvoiceArgumentParser(argparse.ArgumentParser):
    """Argument parser that reserves exit code 2 for manual review."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = InvoiceArgumentParser(
        description="Process one invoice through the invoice automation workflow."
    )
    parser.add_argument(
        "--invoice_path",
        "--invoice-path",
        dest="invoice_path",
        required=True,
        type=Path,
        help="Path to the invoice file to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show technical workflow details and audit events.",
    )
    return parser


def _resolve_invoice_path(invoice_path: Path) -> Path:
    invoice_path = invoice_path.expanduser()

    if not invoice_path.is_absolute():
        invoice_path = PROJECT_ROOT / invoice_path

    return invoice_path.resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _print_startup_failure(
    invoice_path: Path,
    exc: Exception,
    *,
    verbose: bool,
) -> None:
    print("Invoice Processing Result", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    print(file=sys.stderr)
    print("Outcome:   FAILED", file=sys.stderr)
    print(f"Invoice:   {_display_path(invoice_path)}", file=sys.stderr)
    print(
        f"Error:     {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )

    if verbose:
        print("\nTechnical traceback", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        traceback.print_exc()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    invoice_path = _resolve_invoice_path(args.invoice_path)

    if not invoice_path.is_file():
        print(
            f"Error: invoice file not found: {_display_path(invoice_path)}",
            file=sys.stderr,
        )
        return 1

    captured_stdout = io.StringIO()

    try:
        settings = LLMSettings.from_env()
        initialize_database(DATABASE_PATH)

        started_at = perf_counter()

        with redirect_stdout(captured_stdout):
            model = create_chat_model(settings)

            ingestion_agent = IngestionAgent(
                model,
                max_attempts=2,
            )
            approval_agent = ApprovalAgent(
                model,
                max_attempts=2,
            )
            approval_critic = ApprovalCritic(
                model,
                max_attempts=2,
            )

            workflow = build_invoice_workflow(
                ingestion_agent,
                approval_agent,
                approval_critic,
                database_path=DATABASE_PATH,
                max_approval_revisions=2,
                rejection_log_path=REJECTION_LOG_PATH,
                manual_review_log_path=MANUAL_REVIEW_LOG_PATH,
            )

            result = run_invoice_workflow(
                workflow,
                invoice_path,
            )

        elapsed_seconds = perf_counter() - started_at

    except Exception as exc:
        _print_startup_failure(
            invoice_path,
            exc,
            verbose=args.verbose,
        )
        return 1

    output = format_cli_output(
        result,
        verbose=args.verbose,
        provider=settings.provider,
        model=settings.model,
        invoice_path=_display_path(invoice_path),
        rejection_log_path="data/rejections.jsonl",
        manual_review_log_path="data/manual_reviews.jsonl",
        elapsed_seconds=elapsed_seconds,
        captured_stdout=(
            captured_stdout.getvalue()
            if args.verbose
            else None
        ),
    )

    print(output)
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
