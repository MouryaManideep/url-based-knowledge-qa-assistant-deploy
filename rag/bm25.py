"""A small Okapi BM25 keyword index for hybrid retrieval (exact names, codes, numbers)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its me my no not of on or our
    she so that the their them then there these they this to was we were what when where which who why will with
    you your how does do did can could should would about than also been being any all""".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: Dict[str, float] = {}
        self._norm = np.zeros(0, dtype=np.float32)
        self.size = 0

    def fit(self, term_counts: Sequence[Counter]) -> None:
        self.size = len(term_counts)
        if not self.size:
            self._postings, self._idf, self._norm = {}, {}, np.zeros(0, dtype=np.float32)
            return
        lengths = np.array([sum(c.values()) for c in term_counts], dtype=np.float32)
        avg = float(lengths.mean()) or 1.0
        self._norm = self.k1 * (1 - self.b + self.b * lengths / avg)

        postings: Dict[str, Tuple[List[int], List[int]]] = {}
        for doc_id, counts in enumerate(term_counts):
            for term, freq in counts.items():
                ids, freqs = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                freqs.append(freq)
        self._postings = {
            term: (np.array(ids, dtype=np.int64), np.array(freqs, dtype=np.float32))
            for term, (ids, freqs) in postings.items()
        }
        self._idf = {
            term: math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self._postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, freqs = posting
            result[ids] += self._idf[term] * freqs * (self.k1 + 1) / (freqs + self._norm[ids])
        return result
