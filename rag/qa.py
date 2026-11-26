"""Question answering: build the grounded prompt, stream the answer, map citations to sources."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

from config import HISTORY_TURNS
from rag.prompts import HYBRID_SYSTEM, NOT_FOUND, QUESTION_TEMPLATE, STRICT_SYSTEM
from rag.types import Hit

_CITATION = re.compile(r"\[(\d{1,2})\]")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_IMG = re.compile(r"<\s*img\b[^>]*>", re.I)


def retrieval_queries(question: str, history: Sequence[Dict[str, Any]]) -> List[str]:
    """Search with the question, plus the question joined to the previous one for follow-ups
    such as "what about its cost?" whose meaning depends on earlier turns."""
    queries = [question]
    previous = [m for m in history if m.get("role") == "user"]
    if previous:
        queries.append(f"{previous[-1].get('content', '')}\n{question}")
    return queries


def _escape_source_text(text: str) -> str:
    return re.sub(r"</?\s*sources?\b", lambda m: m.group(0).replace("<", "‹"), text, flags=re.I)


def format_context(hits: Sequence[Hit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        label = chunk.title + (f" — {chunk.location}" if chunk.location else "")
        blocks.append(f"[{number}] {_escape_source_text(label)}\n{_escape_source_text(chunk.text)}")
    return "\n\n".join(blocks)


def build_request(
    question: str, hits: Sequence[Hit], history: Sequence[Dict[str, Any]], strict: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (system_instruction, contents) in google-genai format."""
    contents: List[Dict[str, Any]] = []
    for message in list(history)[-HISTORY_TURNS * 2:]:
        role = "user" if message.get("role") == "user" else "model"
        text = str(message.get("content") or "")[:2000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    prompt = QUESTION_TEMPLATE.format(context=format_context(hits), question=question)
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    return (STRICT_SYSTEM if strict else HYBRID_SYSTEM), contents


def sanitize_markdown(text: str) -> str:
    """Remove images from model output: a poisoned page could otherwise make the model emit an
    image URL that leaks conversation data when the browser loads it."""
    return _HTML_IMG.sub("", _MD_IMAGE.sub("", text))


def cited_numbers(answer: str, available: int) -> Set[int]:
    return {int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= available}


def is_not_found(answer: str) -> bool:
    return answer.strip().startswith(NOT_FOUND[:-1])


def hits_to_citations(hits: Sequence[Hit]) -> List[Dict[str, Any]]:
    return [
        {
            "n": number,
            "title": hit.chunk.title,
            "location": hit.chunk.location,
            "url": hit.chunk.url,
            "page": hit.chunk.page,
            "relevance": round(hit.relevance, 3),
            "keyword": hit.keyword_match,
            "snippet": hit.chunk.text[:600],
        }
        for number, hit in enumerate(hits, start=1)
    ]


def stream_answer(llm: Any, question: str, hits: Sequence[Hit], history: Sequence[Dict[str, Any]], strict: bool = True) -> Iterator[str]:
    system, contents = build_request(question, hits, history, strict)
    yield from llm.stream(system, contents)
