"""Source-balanced summaries with structured JSON output."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.index import KnowledgeBase
from rag.llm import LLMError
from rag.prompts import SUMMARY_SYSTEM
from rag.qa import sanitize_markdown
from rag.types import Chunk

# Enough for a representative sample; larger prompts are slower and fail more often under load.
MAX_SUMMARY_CHARS = 24_000


def representative_chunks(kb: KnowledgeBase, source_id: Optional[str] = None, budget: int = MAX_SUMMARY_CHARS) -> List[Chunk]:
    """Pick chunks round-robin across sources, spread evenly through each document, within a budget."""
    by_source: Dict[str, List[Chunk]] = {}
    for chunk in kb.chunks_for(source_id):
        by_source.setdefault(chunk.source_id, []).append(chunk)
    if not by_source:
        return []

    share = budget // len(by_source)
    picked: List[Chunk] = []
    for chunks in by_source.values():
        total = sum(len(c.text) for c in chunks)
        if total <= share:
            picked.extend(chunks)
            continue
        # Even sampling keeps the beginning, middle and end of long documents represented.
        count = max(1, share // max(1, total // len(chunks)))
        step = len(chunks) / count
        picked.extend(chunks[int(i * step)] for i in range(count))
    return picked


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [sanitize_markdown(str(v)).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [line.lstrip("-•* ").strip() for line in value.splitlines() if line.strip()]
    return []


def summarize(llm: Any, kb: KnowledgeBase, source_id: Optional[str] = None) -> Dict[str, Any]:
    chunks = representative_chunks(kb, source_id)
    if not chunks:
        raise LLMError("There is no content to summarize yet.")
    blocks = [f"[{c.title} — {c.location}]\n{c.text}" for c in chunks]
    prompt = "Summarize these source passages:\n\n<sources>\n" + "\n\n".join(blocks) + "\n</sources>"
    data = llm.generate_json(SUMMARY_SYSTEM, prompt)

    overview = sanitize_markdown(str(data.get("overview") or "")).strip()
    key_points = _as_list(data.get("key_points"))
    if not overview or not key_points:
        raise LLMError("The model returned an incomplete summary. Please try again.")
    sources = {c.source_id for c in chunks}
    return {
        "overview": overview,
        "key_points": key_points,
        "takeaways": _as_list(data.get("takeaways")),
        "questions": _as_list(data.get("questions"))[:3],
        "coverage": f"Based on {len(chunks)} passages from {len(sources)} source{'s' if len(sources) != 1 else ''}.",
    }
