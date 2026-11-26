"""Turn a URL or an uploaded file into a Source plus its chunks."""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from config import MAX_EXTRACTED_CHARS, MAX_FILE_BYTES, MAX_OCR_BYTES, MAX_OCR_PAGES
from loaders.pdf_loader import load_pdf
from loaders.web_loader import FetchedPage, fetch_url
from preprocessing.chunker import split_text
from preprocessing.cleaner import parse_html
from rag.types import Chunk, NeedsOCR, Source, SourceError
from utils.helpers import content_fingerprint

TEXT_EXTENSIONS = (".txt", ".md", ".markdown")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_PAGE_MARKER = re.compile(r"^\s*=+\s*Page\s+(\d+)\s*=+\s*$", re.I | re.M)

# OCR callable: (file bytes, MIME type) -> transcribed text. Supplied by the app (Gemini) so this
# module stays testable and works without an API key for ordinary text files.
OCR = Callable[[bytes, str], str]


def split_ocr_pages(text: str) -> List[Tuple[int, str]]:
    """Split OCR output on "=== Page N ===" markers; text without markers is page 1."""
    markers = list(_PAGE_MARKER.finditer(text))
    if not markers:
        return [(1, text.strip())] if text.strip() else []
    pages = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[marker.end():end].strip()
        if body:
            pages.append((int(marker.group(1)), body))
    return pages


def ocr_pages(data: bytes, mime_type: str, ocr: Optional[OCR], page_count: int = 1) -> List[Tuple[int, str]]:
    if ocr is None:
        raise SourceError("This is a scan or photo with no text layer. Add a Gemini API key to read it with OCR.")
    if len(data) > MAX_OCR_BYTES:
        raise SourceError("This scan is too large to read with OCR (18 MB limit). Try splitting it.")
    if page_count > MAX_OCR_PAGES:
        raise SourceError(f"This scan has {page_count} pages; OCR is limited to {MAX_OCR_PAGES}. Try splitting it.")
    pages = split_ocr_pages(ocr(data, mime_type))
    if not pages:
        raise SourceError("OCR found no readable text in this file.")
    if sum(len(t) for _, t in pages) > MAX_EXTRACTED_CHARS:
        raise SourceError("The file contains more text than this app can index in one go.")
    return pages


def normalize_url(url: str) -> str:
    """Canonical form used to detect duplicates: lowercase host, no fragment or trailing slash."""
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/") or ""
    return urlunparse((parsed.scheme.lower(), (parsed.netloc or "").lower(), path, "", parsed.query, ""))


def url_source_id(url: str) -> str:
    return "url-" + content_fingerprint(normalize_url(url).encode())[:16]


def file_source_id(data: bytes) -> str:
    return "file-" + content_fingerprint(data)[:16]


def _chunks_from_sections(
    sections: List[Tuple[str, str]], *, source_id: str, title: str, url: Optional[str], size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for heading, text in sections:
        for piece in split_text(text, size, overlap):
            chunks.append(Chunk(text=piece, source_id=source_id, title=title, location=heading, url=url))
    return chunks


def _chunks_from_pages(
    pages: List[Tuple[int, str]], *, source_id: str, title: str, url: Optional[str], size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for number, text in pages:
        for piece in split_text(text, size, overlap):
            chunks.append(Chunk(text=piece, source_id=source_id, title=title, location=f"Page {number}", url=url, page=number))
    return chunks


def _decode_text(data: bytes, encoding: Optional[str] = None) -> str:
    for candidate in filter(None, [encoding, "utf-8"]):
        try:
            return data.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def ingest_url(url: str, chunk_size: int, chunk_overlap: int, ocr: Optional[OCR] = None) -> Tuple[Source, List[Chunk]]:
    page: FetchedPage = fetch_url(url)
    source_id = url_source_id(url)
    host = urlparse(page.url).hostname or page.url

    if page.kind == "pdf":
        name = page.url.rsplit("/", 1)[-1] or host
        try:
            title, pages = load_pdf(page.body, name)
        except NeedsOCR as scan:
            title, pages = name, ocr_pages(page.body, "application/pdf", ocr, scan.page_count)
        chunks = _chunks_from_pages(pages, source_id=source_id, title=title, url=page.url, size=chunk_size, overlap=chunk_overlap)
        kind = "pdf"
    elif page.kind == "text":
        text = _decode_text(page.body, page.encoding)[:MAX_EXTRACTED_CHARS]
        title = page.url.rsplit("/", 1)[-1] or host
        chunks = _chunks_from_sections([("Content", text)], source_id=source_id, title=title, url=page.url, size=chunk_size, overlap=chunk_overlap)
        kind = "web"
    else:
        title, sections = parse_html(page.body, page.encoding)
        title = title or host
        if sum(len(t) for _, t in sections) > MAX_EXTRACTED_CHARS:
            raise SourceError("The page contains more text than this app can index in one go.")
        chunks = _chunks_from_sections(sections, source_id=source_id, title=title, url=page.url, size=chunk_size, overlap=chunk_overlap)
        kind = "web"

    if not chunks:
        raise SourceError("No readable text found (the page may need JavaScript or a login).")
    return Source(id=source_id, title=title[:200], kind=kind, origin=page.url), chunks


def ingest_file(
    name: str, data: bytes, chunk_size: int, chunk_overlap: int, ocr: Optional[OCR] = None
) -> Tuple[Source, List[Chunk]]:
    if len(data) > MAX_FILE_BYTES:
        raise SourceError("The file is larger than the 25 MB limit.")
    source_id = file_source_id(data)
    lowered = name.lower()
    extension = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    if lowered.endswith(".pdf"):
        try:
            title, pages = load_pdf(data, name)
        except NeedsOCR as scan:
            title, pages = name, ocr_pages(data, "application/pdf", ocr, scan.page_count)
        chunks = _chunks_from_pages(pages, source_id=source_id, title=title, url=None, size=chunk_size, overlap=chunk_overlap)
        kind = "pdf"
    elif extension in IMAGE_TYPES:
        pages = ocr_pages(data, IMAGE_TYPES[extension], ocr)
        text = "\n\n".join(t for _, t in pages)
        chunks = _chunks_from_sections([("Image", text)], source_id=source_id, title=name, url=None, size=chunk_size, overlap=chunk_overlap)
        title, kind = name, "image"
    elif lowered.endswith(TEXT_EXTENSIONS):
        text = _decode_text(data)
        if len(text) > MAX_EXTRACTED_CHARS:
            raise SourceError("The file contains more text than this app can index in one go.")
        title = name
        chunks = _chunks_from_sections([("Content", text)], source_id=source_id, title=title, url=None, size=chunk_size, overlap=chunk_overlap)
        kind = "text"
    else:
        raise SourceError("Unsupported file type. Upload a PDF, an image (PNG, JPG, WebP), or a .txt or .md file.")
    if not chunks:
        raise SourceError("No readable text was found in this file.")
    return Source(id=source_id, title=title[:200], kind=kind, origin=name), chunks
