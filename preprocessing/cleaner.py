"""Turn raw HTML into heading-aware text sections, dropping navigation and boilerplate."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

from bs4 import BeautifulSoup

UNWANTED_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form", "button", "svg", "template"]
_NOISE_WORDS = r"(?:ad|ads|advert|advertisement|advertising|sidebar|navbar|nav-bar|menu|cookie|consent|banner|share|social|newsletter|breadcrumb)"
# Only class/id names that start or end with a noise word ("cookie-banner", "site-sidebar"). Long
# compound names such as Wikipedia's "vector-feature-main-menu-pinned-disabled" are state flags.
NOISE_TOKEN = re.compile(rf"^{_NOISE_WORDS}(?:[-_]|$)|[-_]{_NOISE_WORDS}$", re.I)
PROTECTED_TAGS = {"html", "body", "main", "article"}
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "table", "dt", "dd"]


def _section_text(element) -> str:
    if element.name == "table":
        rows = []
        for row in element.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    if element.name == "pre":
        return element.get_text("\n", strip=True)
    text = element.get_text(" ", strip=True)
    return f"- {text}" if element.name == "li" and text else text


def parse_html(html: Union[str, bytes], encoding: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (page_title, [(heading, text), ...]) with consecutive paragraphs grouped per heading."""
    title, sections = _parse(html, encoding, strip_noise=True)
    if not sections:  # The noise filter was too aggressive for this page; retry without it.
        title, sections = _parse(html, encoding, strip_noise=False)
    return title, sections


def _parse(html: Union[str, bytes], encoding: Optional[str], strip_noise: bool) -> Tuple[str, List[Tuple[str, str]]]:
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None)
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup(UNWANTED_TAGS):
        tag.decompose()
    for tag in list(soup.find_all(True)) if strip_noise else []:
        if tag.decomposed or tag.name in PROTECTED_TAGS:  # Already removed, or a page-level container.
            continue
        attrs = getattr(tag, "attrs", None) or {}
        tokens = list(attrs.get("class", []) or []) + [attrs.get("id", "") or "", attrs.get("role", "") or ""]
        if any(NOISE_TOKEN.search(str(token)) for token in tokens) or attrs.get("aria-hidden") == "true":
            tag.decompose()

    body = soup.body or soup
    root = body
    for candidate in (soup.find("main"), soup.find("article"), soup.find(attrs={"role": "main"})):
        if candidate is not None and len(candidate.get_text(" ", strip=True)) >= 200:
            root = candidate
            break

    sections: List[Tuple[str, List[str]]] = []
    heading = "Introduction"
    for element in root.find_all(BLOCK_TAGS):
        # Nested blocks are already included in their parent's text.
        if element.find_parent(["li", "table", "pre", "blockquote"]):
            continue
        text = _section_text(element)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = text[:200]
            continue
        if sections and sections[-1][0] == heading:
            sections[-1][1].append(text)
        else:
            sections.append((heading, [text]))

    result = [(h, "\n\n".join(parts)) for h, parts in sections]
    if not result:
        text = re.sub(r"\s+", " ", body.get_text(" ", strip=True)).strip()
        if text:
            result = [("Content", text)]
    return title, result
