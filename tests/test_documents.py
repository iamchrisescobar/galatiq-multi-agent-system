from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from invoice_system.documents import (
    DocumentLoadError,
    load_document,
)


@pytest.mark.parametrize(
    ("extension", "content", "expected_media_type"),
    [
        (
            ".txt",
            "Vendor: Example Vendor\nTotal: $100.00",
            "text/plain",
        ),
        (
            ".json",
            '{"vendor": "Example Vendor", "total": 100}',
            "application/json",
        ),
        (
            ".csv",
            "field,value\nvendor,Example Vendor\ntotal,100",
            "text/csv",
        ),
        (
            ".xml",
            "<invoice><vendor>Example Vendor</vendor></invoice>",
            "application/xml",
        ),
    ],
)
def test_load_document_reads_text_based_formats(
    tmp_path: Path,
    extension: str,
    content: str,
    expected_media_type: str,
) -> None:
    invoice_path = tmp_path / f"invoice{extension}"
    invoice_path.write_text(content, encoding="utf-8")

    result = load_document(invoice_path)

    assert result.path == invoice_path.resolve()
    assert result.text == content
    assert result.media_type == expected_media_type
    assert result.page_count is None


def test_load_document_extracts_pdf_text(tmp_path: Path) -> None:
    invoice_path = tmp_path / "invoice.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "INVOICE\nVendor: Example Vendor\nTotal: $100.00",
    )
    document.save(str(invoice_path))
    document.close()

    result = load_document(invoice_path)

    assert result.path == invoice_path.resolve()
    assert result.media_type == "application/pdf"
    assert result.page_count == 1
    assert "[Page 1]" in result.text
    assert "Example Vendor" in result.text
    assert "$100.00" in result.text


def test_load_document_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(
        DocumentLoadError,
        match="does not exist",
    ):
        load_document(missing_path)


def test_load_document_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(
        DocumentLoadError,
        match="not a file",
    ):
        load_document(tmp_path)


def test_load_document_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    invoice_path = tmp_path / "invoice.docx"
    invoice_path.write_text("Example", encoding="utf-8")

    with pytest.raises(
        DocumentLoadError,
        match="Unsupported invoice format",
    ):
        load_document(invoice_path)


def test_load_document_rejects_empty_text_file(
    tmp_path: Path,
) -> None:
    invoice_path = tmp_path / "empty.txt"
    invoice_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(
        DocumentLoadError,
        match="no readable text",
    ):
        load_document(invoice_path)


def test_load_document_rejects_pdf_without_extractable_text(
    tmp_path: Path,
) -> None:
    invoice_path = tmp_path / "blank.pdf"

    document = pymupdf.open()
    document.new_page()
    document.save(str(invoice_path))
    document.close()

    with pytest.raises(
        DocumentLoadError,
        match="no readable text",
    ):
        load_document(invoice_path)