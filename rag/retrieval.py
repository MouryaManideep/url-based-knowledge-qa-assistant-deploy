"""Dense semantic retrieval helpers for the FAISS knowledge base."""
from __future__ import annotations

from copy import copy
from typing import Any, List


def _cosine_similarity_from_l2_distance(distance: float) -> float:
    """Convert squared L2 distance to cosine similarity for unit-normalized vectors."""
    return max(-1.0, min(1.0, 1.0 - float(distance) / 2.0))


def _document_key(document: Any) -> tuple:
    metadata = document.metadata or {}
    return (
        document.page_content,
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("heading"),
    )


def semantic_search(
    db: Any,
    query: str,
    k: int = 5,
    min_similarity: float = 0.20,
    search_type: str = "semantic_similarity",
) -> List[Document]:
    """Retrieve meaning-matched passages, optionally diversified with MMR.

    The current embedding model normalizes vectors, and the FAISS index uses
    squared L2 distance. For unit vectors, cosine similarity is 1 - d^2 / 2.
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    if k < 1:
        raise ValueError("k must be at least 1")

    fetch_k = max(k, min(30, k * 4))
    candidates = db.similarity_search_with_score(query, k=fetch_k if search_type == "mmr" else k)
    scores = {_document_key(doc): _cosine_similarity_from_l2_distance(distance) for doc, distance in candidates}

    if search_type == "mmr":
        selected = db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    else:
        selected = [doc for doc, _ in candidates]

    results = []
    for document in selected:
        score = scores.get(_document_key(document))
        if score is None or score < min_similarity:
            continue
        scored_document = copy(document)
        scored_document.metadata = {**document.metadata, "semantic_similarity": round(score, 4)}
        results.append(scored_document)
    return results
