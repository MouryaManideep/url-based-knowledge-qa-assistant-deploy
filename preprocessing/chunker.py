"""Recursive, overlap-aware text splitting (no LangChain dependency)."""
from __future__ import annotations

import re
from typing import List

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_recursive(text: str, size: int, separators: List[str]) -> List[str]:
    if len(text) <= size:
        return [text]
    for index, separator in enumerate(separators):
        if separator in text:
            parts = text.split(separator)
            pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
            result: List[str] = []
            for piece in pieces:
                if not piece:
                    continue
                if len(piece) <= size:
                    result.append(piece)
                else:
                    result.extend(_split_recursive(piece, size, separators[index + 1:]))
            return result
    return [text[i:i + size] for i in range(0, len(text), size)]


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Split text into chunks of at most chunk_size characters, overlapping by up to chunk_overlap."""
    if not isinstance(chunk_size, int) or chunk_size < 50:
        raise ValueError("chunk_size must be an integer of at least 50")
    if not isinstance(chunk_overlap, int) or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be a non-negative integer smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    pieces = _split_recursive(text, chunk_size, SEPARATORS)
    chunks: List[str] = []
    window: List[str] = []
    length = 0
    for piece in pieces:
        if window and length + len(piece) > chunk_size:
            chunks.append("".join(window).strip())
            # Keep a tail of the previous chunk as overlap, as long as the new piece still fits.
            while window and (length > chunk_overlap or length + len(piece) > chunk_size):
                length -= len(window[0])
                window.pop(0)
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [chunk for chunk in chunks if chunk]
