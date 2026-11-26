"""Local embeddings via FastEmbed (ONNX Runtime): no PyTorch, small install, runs on CPU."""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

from config import EMBEDDING_MODEL, QUERY_INSTRUCTION


class Embedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        return matrix / max(float(np.linalg.norm(matrix)), 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class FastEmbedder:
    """Wraps fastembed.TextEmbedding. The model (~70 MB) downloads on first use and is cached."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, query_instruction: str = QUERY_INSTRUCTION):
        from fastembed import TextEmbedding  # Imported lazily so the app starts instantly.

        self.model_name = model_name
        self.query_instruction = query_instruction if "bge" in model_name.lower() else ""
        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[np.ndarray] = list(self._model.embed(list(texts), batch_size=batch_size))
        return normalize_rows(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.embed([self.query_instruction + text])))
        return normalize_rows(np.asarray(vector))
