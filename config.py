"""Central settings. Values can be overridden with environment variables or Streamlit secrets."""
from __future__ import annotations

import os
from typing import Optional


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting from Streamlit secrets first, then the environment."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value is not None else default
    except Exception:  # No secrets file, or Streamlit not running.
        pass
    return os.getenv(name, default)


def _int_setting(name: str, default: int) -> int:
    try:
        return int(get_setting(name, str(default)))
    except (TypeError, ValueError):
        return default


# Models
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order when the primary model is overloaded (503), out of quota (429) or unavailable (404).
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in (get_setting("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite") or "").split(",") if m.strip()
]
GEMINI_MODELS = list(dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]))
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models retrieve better when queries carry this instruction; passages are embedded as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Per-session limits keep one visitor from exhausting a shared host.
MAX_SOURCES_PER_SESSION = _int_setting("MAX_SOURCES_PER_SESSION", 25)
MAX_CHUNKS_PER_SESSION = _int_setting("MAX_CHUNKS_PER_SESSION", 4000)
MAX_QUESTION_CHARS = 2000

# Ingestion limits
MAX_WEB_BYTES = 5 * 1024 * 1024
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_EXTRACTED_CHARS = 3_000_000

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Retrieval defaults
DEFAULT_TOP_K = 6
DEFAULT_MIN_RELEVANCE = 0.30
HISTORY_TURNS = 6

SAMPLE_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
