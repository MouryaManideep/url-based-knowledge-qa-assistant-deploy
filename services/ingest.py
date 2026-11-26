"""Turn a URL or an uploaded file into a Source plus its chunks."""
from __future__ import annotations

from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from config import MAX_EXTRACTED_CHARS, MAX_FILE_BYTES
from loaders.pdf_loader import load_pdf
from loaders.web_loader import FetchedPage, fetch_url
from preprocessing.chunker import split_text
from preprocessing.cleaner import parse_html
from rag.types import Chunk, Source, SourceError
from utils.helpers import content_fingerprint

TEXT_EXTENSIONS = (".txt", ".md", ".markdown")


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


def ingest_url(url: str, chunk_size: int, chunk_overlap: int) -> Tuple[Source, List[Chunk]]:
    page: FetchedPage = fetch_url(url)
    source_id = url_source_id(url)
    host = urlparse(page.url).hostname or page.url

    if page.kind == "pdf":
        title, pages = load_pdf(page.body, page.url.rsplit("/", 1)[-1] or host)
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


def ingest_file(name: str, data: bytes, chunk_size: int, chunk_overlap: int) -> Tuple[Source, List[Chunk]]:
    if len(data) > MAX_FILE_BYTES:
        raise SourceError("The file is larger than the 25 MB limit.")
    source_id = file_source_id(data)
    lowered = name.lower()
    if lowered.endswith(".pdf"):
        title, pages = load_pdf(data, name)
        chunks = _chunks_from_pages(pages, source_id=source_id, title=title, url=None, size=chunk_size, overlap=chunk_overlap)
        kind = "pdf"
    elif lowered.endswith(TEXT_EXTENSIONS):
        text = _decode_text(data)
        if len(text) > MAX_EXTRACTED_CHARS:
            raise SourceError("The file contains more text than this app can index in one go.")
        title = name
        chunks = _chunks_from_sections([("Content", text)], source_id=source_id, title=title, url=None, size=chunk_size, overlap=chunk_overlap)
        kind = "text"
    else:
        raise SourceError("Unsupported file type. Upload a PDF, .txt or .md file.")
    if not chunks:
        raise SourceError("No readable text was found in this file.")
    return Source(id=source_id, title=title[:200], kind=kind, origin=name), chunks
