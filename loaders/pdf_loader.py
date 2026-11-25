import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import os
from utils.logger import setup_logger
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_EXTRACTED_CHARS = 5_000_000

# File: loaders/pdf_loader.py
# Purpose: Read PDF files and turn each page into a simple text record for the RAG pipeline.

logger = setup_logger("pdf_loader")

def load_pdf_content(file_source: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file path or file-like object (bytes).
    Returns list of dicts: {"page_num": int, "text": str, "title": str, "source": str}
    """
    doc = None
    try:
        if isinstance(file_source, bytes):
            if len(file_source) > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the configured size limit")
            doc = fitz.open(stream=file_source, filetype="pdf")
        elif isinstance(file_source, str) and os.path.exists(file_source):
            if os.path.getsize(file_source) > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the configured size limit")
            doc = fitz.open(file_source)
        else:
            if hasattr(file_source, "read"):
                content = file_source.read(MAX_PDF_BYTES + 1)
                if len(content) > MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds the configured size limit")
                doc = fitz.open(stream=content, filetype="pdf")
            else:
                raise ValueError("Unsupported PDF file source type.")

        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("Password-protected PDFs are not supported")
        if len(doc) > MAX_PDF_PAGES:
            raise ValueError("PDF exceeds the configured page limit")
        title = doc.metadata.get("title") or file_name
        if not title:
            title = "Uploaded PDF"

        logger.info(f"Loading PDF with {len(doc)} pages: {file_name}")
        pages_data = []
        total_chars = 0
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text()
            total_chars += len(text)
            if total_chars > MAX_EXTRACTED_CHARS:
                raise ValueError("PDF extracted text exceeds the configured limit")
            pages_data.append({
                "page_num": page_idx + 1,
                "text": text,
                "title": title,
                "source": file_name
            })
            
        return pages_data
    except Exception as e:
        logger.warning("Could not read PDF (%s).", type(e).__name__)
        return []
    finally:
        if doc is not None:
            doc.close()
