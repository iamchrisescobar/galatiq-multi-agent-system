from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pymupdf


SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".txt",
        ".json",
        ".csv",
        ".xml",
        ".pdf",
    }
)

MEDIA_TYPES: Final[dict[str, str]] = {
    ".txt": "text/plain",
    ".json": "application/json",
    ".csv": "text/csv",
    ".xml": "application/xml",
    ".pdf": "application/pdf",
}


class DocumentLoadError(RuntimeError):
    """Raised when an invoice document cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedDocument:
    """
    Text extracted from an invoice file.

    The loader's only responsibility is to convert supported 
    local files into text for the ingestion agent.
    """

    path: Path
    text: str
    media_type: str
    page_count: int | None = None


def load_document(invoice_path: str | Path) -> LoadedDocument:
    """
    Load a supported invoice file and return its textual content.

    Text-based formats are read without transforming their contents. PDF text
    is extracted page by page and labeled so the downstream ingestion agent
    retains document structure.
    """

    path = Path(invoice_path).expanduser()

    if not path.exists():
        raise DocumentLoadError(
            f"Invoice document does not exist: {path}"
        )

    if not path.is_file():
        raise DocumentLoadError(
            f"Invoice document path is not a file: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(
            f"Unsupported invoice format {extension!r}. "
            f"Supported formats: {supported}."
        )

    resolved_path = path.resolve()

    if extension == ".pdf":
        text, page_count = _load_pdf(resolved_path)
    else:
        text = _load_text_file(resolved_path)
        page_count = None

    normalized_text = text.strip()

    if not normalized_text:
        raise DocumentLoadError(
            f"Invoice document contains no readable text: {resolved_path}"
        )

    return LoadedDocument(
        path=resolved_path,
        text=normalized_text,
        media_type=MEDIA_TYPES[extension],
        page_count=page_count,
    )


def _load_text_file(path: Path) -> str:
    """
    Read a UTF-8 text-based document.

    utf-8-sig handles files that begin with a UTF-8 byte-order mark.
    """

    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Invoice document is not valid UTF-8 text: {path}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"Could not read invoice document: {path}"
        ) from exc


def _load_pdf(path: Path) -> tuple[str, int]:
    """
    Extract text from each page of a PDF.

    Image-only PDFs fail on purpose with a clear message. Considering OCR as 
    beyond the initial MVP and can be added later as a fallback.
    """

    try:
        with pymupdf.open(str(path)) as document:
            page_count = document.page_count
            page_sections: list[str] = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()

                if not page_text:
                    continue

                page_sections.append(
                    f"[Page {page_number}]\n{page_text}"
                )

    except (OSError, RuntimeError, ValueError) as exc:
        raise DocumentLoadError(
            f"Could not extract text from PDF: {path}"
        ) from exc

    return "\n\n".join(page_sections), page_count