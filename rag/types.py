"""Plain data types shared across the pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


class SourceError(Exception):
    """A source could not be ingested. The message is safe to show to users."""


class NeedsOCR(SourceError):
    """The file has no text layer (a scan or a photo); its text can only be read with OCR."""

    def __init__(self, message: str, page_count: int = 1):
        super().__init__(message)
        self.page_count = page_count


@dataclass(frozen=True)
class Chunk:
    text: str
    source_id: str
    title: str
    location: str = ""
    url: Optional[str] = None
    page: Optional[int] = None

    @property
    def embed_text(self) -> str:
        """Text used for embedding and keyword search: a contextual header plus the passage."""
        header = self.title
        if self.location and self.location.lower() not in {"introduction", "content"}:
            header = f"{header} — {self.location}"
        return f"{header}\n{self.text}"


@dataclass
class Source:
    id: str
    title: str
    kind: str  # "web", "pdf" or "text"
    origin: str  # URL or file name
    chunk_count: int = 0
    char_count: int = 0
    added_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    relevance: float  # cosine similarity of the best-matching query, 0..1
    keyword_match: bool = False
