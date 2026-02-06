"""
OpenAI Embedding 模組
負責將文本轉換為向量
支持本地緩存以節省 API 調用成本
"""

import os
from typing import List, Optional
from openai import OpenAI
import tiktoken

from .embedding_cache import EmbeddingCache


class Embedder:
    """OpenAI Embedding wrapper with caching"""

    def __init__(self, model: str = None, use_cache: bool = True, cache_dir: str = None):
        """
        Args:
            model: embedding 模型名稱
            use_cache: 是否啟用本地緩存
            cache_dir: 緩存目錄路徑
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        # 緩存
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir) if use_cache else None

        # 統計
        self.api_calls = 0
        self.cached_results = 0

    def get_dimension(self) -> int:
        """獲取當前模型的向量維度"""
        return self.dimensions.get(self.model, 1536)

    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        return len(self.encoding.encode(text))

    def embed_text(self, text: str) -> List[float]:
        """將單個文本轉換為向量

        Args:
            text: 文本內容

        Returns:
            embedding 向量
        """
        # 檢查緩存
        if self.use_cache and self.cache:
            cached = self.cache.get(text, self.model)
            if cached is not None:
                self.cached_results += 1
                return cached

        # 調用 API
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding = response.data[0].embedding
        self.api_calls += 1

        # 保存到緩存
        if self.use_cache and self.cache:
            self.cache.set(text, embedding, self.model)

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """批量將文本轉換為向量

        Args:
            texts: 文本列表
            batch_size: 每批處理的數量 (OpenAI 限制)

        Returns:
            向量列表
        """
        if not texts:
            return []

        all_embeddings = [None] * len(texts)

        # 檢查緩存
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

        # 調用 API 獲取未緩存的 embeddings
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_indices = indices_to_embed[i:i + batch_size]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                self.api_calls += 1

                for j, item in enumerate(response.data):
                    embedding = item.embedding
                    original_index = batch_indices[j]
                    all_embeddings[original_index] = embedding

                    # 保存到緩存
                    if self.use_cache and self.cache:
                        self.cache.set(batch[j], embedding, self.model)

        # 保存緩存索引
        if self.use_cache and self.cache:
            self.cache.save()

        return all_embeddings

    def get_stats(self) -> dict:
        """獲取統計信息"""
        stats = {
            "model": self.model,
            "api_calls": self.api_calls,
            "cached_results": self.cached_results,
            "cache_enabled": self.use_cache
        }
        if self.use_cache and self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        return stats


if __name__ == "__main__":
    # 測試
    from dotenv import load_dotenv
    load_dotenv()

    embedder = Embedder(use_cache=True)

    # 測試單個文本
    text = "OptimusPay is a payment processing platform"
    embedding = embedder.embed_text(text)
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Token count: {embedder.count_tokens(text)}")

    # 第二次調用應該從緩存獲取
    embedding2 = embedder.embed_text(text)
    print(f"\nSecond call (should be cached): dimension={len(embedding2)}")

    # 測試批量
    texts = [
        "Deposit transaction processing",
        "Payout bank transfer",
        "Robot automation for banking"
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embeddings: {len(embeddings)} vectors")

    # 統計
    print(f"\nStats: {embedder.get_stats()}")
