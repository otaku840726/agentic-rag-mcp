"""
BM25 Tokenizer — lightweight client-side BM25 sparse vector generator.

Produces SparseVector (same interface as SparseEmbedder) using pure string
operations.  No neural model, no inference — just regex tokenization +
deterministic hashing.

Qdrant applies server-side IDF weighting via Modifier.IDF on the collection,
so we only need to supply raw term-frequency (TF) values here.
"""

import re
import hashlib
from collections import Counter
from typing import List

from .sparse_embedder import SparseVector


class Bm25Tokenizer:
    """Client-side BM25 tokenizer that produces SparseVector objects."""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self._token_re = re.compile(r"\w+")

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase regex tokenization."""
        return self._token_re.findall(text.lower())

    def _token_to_index(self, token: str) -> int:
        """Deterministic token → index via MD5 hash (not affected by PYTHONHASHSEED)."""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.vocab_size

    def embed_text(self, text: str) -> SparseVector:
        """Produce a sparse TF vector for a single text."""
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf = Counter(self._token_to_index(t) for t in tokens)
        indices = sorted(tf.keys())
        values = [float(tf[i]) for i in indices]
        return SparseVector(indices=indices, values=values)

    def embed_batch(self, texts: List[str]) -> List[SparseVector]:
        """Produce sparse TF vectors for a batch of texts."""
        return [self.embed_text(t) for t in texts]
