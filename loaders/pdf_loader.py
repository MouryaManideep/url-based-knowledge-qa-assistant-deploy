"""Extract text from PDFs page by page with pypdf (pure Python, permissive license)."""
from __future__ import annotations

import io
from typing import List, Tuple

from pypdf import PdfReader

from config import MAX_EXTRACTED_CHARS, MAX_FILE_BYTES, MAX_PDF_PAGES
from rag.types import NeedsOCR, SourceError
from utils.logger import setup_logger

logger = setup_logger("pdf_loader")


def load_pdf(data: bytes, file_name: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Return (title, [(page_number, text), ...]). Raises SourceError on unreadable input."""
    if len(data) > MAX_FILE_BYTES:
        raise SourceError("The PDF is larger than the 25 MB limit.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise SourceError("Password-protected PDFs are not supported.")
        page_count = len(reader.pages)
    except SourceError:
        raise
    except Exception as exc:
        logger.warning("Could not open PDF (%s).", type(exc).__name__)
        raise SourceError("This file is not a readable PDF.") from exc

    if page_count > MAX_PDF_PAGES:
        raise SourceError(f"The PDF has {page_count} pages; the limit is {MAX_PDF_PAGES}.")

    title = file_name
    try:
        meta_title = (reader.metadata or {}).get("/Title")
        if meta_title and str(meta_title).strip():
            title = str(meta_title).strip()
    except Exception:
        pass

    pages: List[Tuple[int, str]] = []
    total = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # One broken page should not sink the document.
            logger.warning("Skipped unreadable PDF page %s (%s).", number, type(exc).__name__)
            continue
        total += len(text)
        if total > MAX_EXTRACTED_CHARS:
            raise SourceError("The PDF contains more text than this app can index in one go.")
        if text.strip():
            pages.append((number, text))

    if not pages:
        raise NeedsOCR("This PDF is a scan with no text layer.", page_count=page_count)
    return title, pages
