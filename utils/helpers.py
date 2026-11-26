"""Shared helpers: URL validation, fingerprints, Markdown escaping and chat export."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import quote, urlparse


def is_valid_url(url: str) -> bool:
    """Validate HTTP(S) URL syntax; the loader separately checks network safety."""
    try:
        result = urlparse(str(url).strip())
        return (
            result.scheme.lower() in {"http", "https"}
            and bool(result.hostname)
            and not re.search(r"\s", result.hostname)
            and not result.username
            and not result.password
            and (result.port is None or 1 <= result.port <= 65535)
        )
    except (AttributeError, ValueError):
        return False


def content_fingerprint(data: bytes) -> str:
    """Return a stable SHA-256 fingerprint for binary contents."""
    return hashlib.sha256(data).hexdigest()


def parse_url_list(text: str) -> List[str]:
    """Split pasted text on whitespace/commas, keep order, drop duplicates."""
    seen, result = set(), []
    for token in re.split(r"[\s,]+", text or ""):
        token = token.strip().strip("<>\"'")
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def md_escape(value: Any) -> str:
    """Escape untrusted text for inline use inside Streamlit Markdown."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", text)


def safe_link(url: Any) -> str:
    return quote(str(url), safe=":/?#[]@!$&'*,;=%+~-._")


def export_chat_to_markdown(history: List[Dict[str, Any]]) -> str:
    lines = [f"# Chat export", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for message in history:
        role = "You" if message.get("role") == "user" else "Assistant"
        lines += [f"### {role}", str(message.get("content", "")), ""]
        citations = message.get("citations") or []
        cited = set(message.get("cited") or [])
        shown = [c for c in citations if not cited or c.get("n") in cited]
        if shown:
            lines.append("**Sources**")
            for cite in shown:
                label = f"[{cite.get('n')}] {md_escape(cite.get('title', 'Untitled'))}"
                if cite.get("location"):
                    label += f" — {md_escape(cite['location'])}"
                url = cite.get("url")
                lines.append(f"- {label} ({safe_link(url)})" if url and is_valid_url(url) else f"- {label}")
            lines.append("")
    return "\n".join(lines)
