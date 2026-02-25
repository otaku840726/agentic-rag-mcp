"""
Embedding 模組
負責將文本轉換為向量，支持任意 OpenAI-compatible provider
"""

import logging
import time
from typing import List, Optional

from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

from .embedding_cache import EmbeddingCache


class Embedder:
    """Embedding wrapper with caching — provider/model 從 config.yaml 讀取"""

    def __init__(self, use_cache: bool = True, cache_dir: str = None):
        from ..provider import create_client, get_component_config, load_config

        cfg = get_component_config("embedding")
        self.client: OpenAI = create_client(cfg.provider)
        self.model: str = cfg.model
        self.identifier: str = cfg.identifier or self.model

        # Read batch_size from config (embedding.batch_size)
        full_cfg = load_config()
        self.batch_size: int = int(
            full_cfg.get("embedding", {}).get("batch_size", 100)
        )

        # dimension 首次 embed 時自動偵測
        self._dimension: Optional[int] = None

        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir, identifier=self.identifier) if use_cache else None

        self.api_calls = 0
        self.cached_results = 0

    def get_dimension(self) -> int:
        """取得向量維度（首次呼叫時自動偵測）"""
        if self._dimension is None:
            resp = self.client.embeddings.create(input=["dimension probe"], model=self.model)
            self._dimension = len(resp.data[0].embedding)
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """將單個文本轉換為向量"""
        if self.use_cache and self.cache:
            cached = self.cache.get(text, self.model)
            if cached is not None:
                self.cached_results += 1
                return cached

        response = self._embed_with_retry([text])
        embedding = response.data[0].embedding
        self.api_calls += 1

        if self._dimension is None:
            self._dimension = len(embedding)

        if self.use_cache and self.cache:
            self.cache.set(text, embedding, self.model)

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """批量將文本轉換為向量"""
        if not texts:
            return []

        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        texts_to_embed = []
        indices_to_embed = []

        if self.use_cache and self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model)
                if cached is not None:
                    all_embeddings[i] = cached
                    self.cached_results += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), effective_batch_size):
                batch = texts_to_embed[i:i + effective_batch_size]
                batch_indices = indices_to_embed[i:i + effective_batch_size]

                response = self._embed_with_retry(batch)
                self.api_calls += 1

                for j, item in enumerate(response.data):
                    embedding = item.embedding
                    original_index = batch_indices[j]
                    all_embeddings[original_index] = embedding

                    if self._dimension is None:
                        self._dimension = len(embedding)

                    if self.use_cache and self.cache:
                        self.cache.set(batch[j], embedding, self.model)

            if self.use_cache and self.cache:
                self.cache.save()

        return all_embeddings

    def _embed_with_retry(self, texts: List[str], max_retries: int = 5):
        """Call embeddings API with exponential backoff on 429 rate limit errors."""
        wait = 10
        for attempt in range(max_retries):
            try:
                return self.client.embeddings.create(model=self.model, input=texts)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait)
                wait = min(wait * 2, 120)

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "dimension": self._dimension,
            "api_calls": self.api_calls,
            "cached_results": self.cached_results,
            "cache_enabled": self.use_cache,
        }
