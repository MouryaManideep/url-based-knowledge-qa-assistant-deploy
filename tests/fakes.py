"""Test doubles: a deterministic bag-of-words embedder and a scripted LLM."""
import hashlib

import numpy as np

from rag.bm25 import tokenize


class HashingEmbedder:
    dim = 256

    def _vector(self, text):
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_passages(self, texts):
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


class FakeLLM:
    def __init__(self, deltas=("Answer [1].",), json_payload=None, error=None):
        self.deltas, self.json_payload, self.error = list(deltas), json_payload, error
        self.calls = []

    def stream(self, system, contents):
        self.calls.append((system, contents))
        if self.error:
            raise self.error
        yield from self.deltas

    def generate_json(self, system, prompt):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.json_payload
