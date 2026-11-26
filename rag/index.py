"""Per-session, in-memory knowledge base with hybrid (dense + BM25) retrieval.

Each browser session owns its own KnowledgeBase, so visitors never see each other's
documents. Brute-force cosine search over normalized vectors is exact and, at the
per-session limit of a few thousand chunks, takes well under a millisecond.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import MAX_CHUNKS_PER_SESSION, MAX_SOURCES_PER_SESSION
from rag.bm25 import BM25, tokenize
from rag.embeddings import Embedder, normalize_rows
from rag.types import Chunk, Hit, Source, SourceError

RRF_K = 60


class KnowledgeBase:
    def __init__(self, max_sources: int = MAX_SOURCES_PER_SESSION, max_chunks: int = MAX_CHUNKS_PER_SESSION):
        self.max_sources = max_sources
        self.max_chunks = max_chunks
        self.sources: Dict[str, Source] = {}
        self.chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._terms: List[Counter] = []
        self._bm25 = BM25()

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def has_source(self, source_id: str) -> bool:
        return source_id in self.sources

    def check_capacity(self, new_chunks: int = 0) -> None:
        if len(self.sources) >= self.max_sources:
            raise SourceError(f"You've reached the limit of {self.max_sources} sources. Remove one to add more.")
        if len(self.chunks) + new_chunks > self.max_chunks:
            raise SourceError(
                f"This would exceed the session limit of {self.max_chunks:,} passages. Remove a source or add a smaller one."
            )

    def add(self, source: Source, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if source.id in self.sources:
            raise SourceError("This source is already in your knowledge base.")
        if not chunks:
            raise SourceError("No readable text was found.")
        self.check_capacity(len(chunks))

        vectors = normalize_rows(embedder.embed_passages([c.embed_text for c in chunks]))
        if vectors.shape[0] != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        if self._vectors is not None and self._vectors.shape[1] != vectors.shape[1]:
            raise RuntimeError("Embedding dimension changed within a session")

        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.chunks.extend(chunks)
        self._terms.extend(Counter(tokenize(c.embed_text)) for c in chunks)
        source.chunk_count = len(chunks)
        source.char_count = sum(len(c.text) for c in chunks)
        self.sources[source.id] = source
        self._bm25.fit(self._terms)

    def remove(self, source_id: str) -> None:
        if source_id not in self.sources:
            return
        keep = [i for i, c in enumerate(self.chunks) if c.source_id != source_id]
        self.chunks = [self.chunks[i] for i in keep]
        self._terms = [self._terms[i] for i in keep]
        self._vectors = self._vectors[keep] if (self._vectors is not None and keep) else None
        del self.sources[source_id]
        self._bm25.fit(self._terms)

    def clear(self) -> None:
        self.__init__(self.max_sources, self.max_chunks)

    def chunks_for(self, source_id: Optional[str] = None) -> List[Chunk]:
        return [c for c in self.chunks if source_id is None or c.source_id == source_id]

    # -------------------------------------------------------------- retrieval
    def search(
        self,
        queries: Sequence[str],
        embedder: Embedder,
        k: int = 6,
        min_relevance: float = 0.30,
        diversify: bool = False,
    ) -> List[Hit]:
        """Hybrid search: dense cosine + BM25, fused with Reciprocal Rank Fusion.

        `queries` is usually [question] or [question, question-with-context] for follow-ups.
        A passage is kept if its cosine similarity clears `min_relevance` or it is a top
        keyword match, so exact terms (IDs, names) are not lost to the embedding model.
        """
        queries = [q for q in queries if q and q.strip()]
        if self.is_empty or not queries or k < 1:
            return []
        n = len(self.chunks)
        pool = min(n, max(k * 4, 20))
        fused: Dict[int, float] = {}
        best_dense = np.full(n, -1.0, dtype=np.float32)
        keyword_hits: set = set()

        for query in queries:
            dense = self._vectors @ normalize_rows(embedder.embed_query(query))
            best_dense = np.maximum(best_dense, dense)
            for rank, idx in enumerate(np.argsort(-dense)[:pool]):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

            lexical = self._bm25.scores(query)
            if lexical.any():
                for rank, idx in enumerate(np.argsort(-lexical)[:pool]):
                    if lexical[idx] <= 0:
                        break
                    keyword_hits.add(int(idx))
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        candidates = [i for i in fused if best_dense[i] >= min_relevance or i in keyword_hits]
        candidates.sort(key=lambda i: fused[i], reverse=True)
        if diversify and len(candidates) > k:
            chosen = self._mmr(candidates, fused, k)
        else:
            chosen = candidates[:k]
        return [
            Hit(chunk=self.chunks[i], relevance=float(max(0.0, min(1.0, best_dense[i]))), keyword_match=i in keyword_hits)
            for i in chosen
        ]

    def _mmr(self, candidates: List[int], fused: Dict[int, float], k: int, lam: float = 0.7) -> List[int]:
        """Maximal marginal relevance: trade a little relevance for less redundant passages."""
        top = max(fused[i] for i in candidates)
        relevance = {i: fused[i] / top for i in candidates}
        vectors = self._vectors[candidates]
        similarity = vectors @ vectors.T
        position = {idx: p for p, idx in enumerate(candidates)}
        chosen: List[int] = [candidates[0]]
        remaining = candidates[1:]
        while remaining and len(chosen) < k:
            def score(i: int) -> float:
                redundancy = max(similarity[position[i], position[j]] for j in chosen)
                return lam * relevance[i] - (1 - lam) * float(redundancy)

            best = max(remaining, key=score)
            chosen.append(best)
            remaining.remove(best)
        return chosen
