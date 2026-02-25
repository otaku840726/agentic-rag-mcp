"""
Hybrid Search - Dense + Sparse (BM25) + RRF Fusion
連接 Qdrant 向量數據庫，支持真正的 hybrid search
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector as QdrantSparseVector,
)
import tiktoken

from .provider import load_config, create_client, get_component_config, get_sparse_config


@dataclass
class SearchConfig:
    """搜索配置"""
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str
    embedding_model: str = "text-embedding-3-small"
    top_n: int = 100


class HybridSearch:
    """混合搜索引擎 - 支持 dense, sparse (BM25), hybrid (RRF fusion)"""

    def __init__(self, config: Optional[SearchConfig] = None):
        cfg = load_config()
        qdrant_cfg = cfg.get("qdrant", {})

        self.config = config or SearchConfig(
            qdrant_url=qdrant_cfg.get("url") or os.getenv("QDRANT_URL", ""),
            qdrant_api_key=qdrant_cfg.get("api_key") or os.getenv("QDRANT_API_KEY", ""),
            collection_name=qdrant_cfg.get("collection") or os.getenv("QDRANT_COLLECTION", "codebase"),
        )

        # Embedding component config
        embed_cfg = get_component_config("embedding")
        self.config.embedding_model = embed_cfg.model

        # Qdrant client
        self.qdrant = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key
        )

        # Embedding client (from provider)
        self.openai = create_client(embed_cfg.provider)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Sparse mode from config
        sparse_cfg = get_sparse_config()
        self._sparse_mode = sparse_cfg["mode"]  # qdrant-bm25 | splade | disabled
        self._sparse_cfg = sparse_cfg

        # Lazy-loaded sparse encoder (Bm25Tokenizer or SparseTextEmbedding)
        self._sparse_embedder = None
        self._sparse_support = None  # None = not yet checked

    def _check_sparse_support(self) -> bool:
        """檢測 collection 是否有 sparse vectors (向後兼容)"""
        if self._sparse_support is not None:
            return self._sparse_support

        # Mode explicitly disabled — skip collection check
        if self._sparse_mode == "disabled":
            self._sparse_support = False
            return False

        try:
            info = self.qdrant.get_collection(self.config.collection_name)
            sparse_config = info.config.params.sparse_vectors
            self._sparse_support = bool(sparse_config and "sparse" in sparse_config)
        except Exception:
            self._sparse_support = False

        return self._sparse_support

    def _get_sparse_embedder(self):
        """Lazy-load sparse embedder based on mode."""
        if self._sparse_embedder is None:
            if self._sparse_mode == "qdrant-bm25":
                from .indexer.bm25_tokenizer import Bm25Tokenizer
                vocab_size = int(self._sparse_cfg["bm25"].get("vocab_size", 30000))
                self._sparse_embedder = Bm25Tokenizer(vocab_size=vocab_size)
            elif self._sparse_mode == "splade":
                from fastembed import SparseTextEmbedding
                model_name = self._sparse_cfg["splade"].get("model", "prithivida/Splade_PP_en_v1")
                self._sparse_embedder = SparseTextEmbedding(model_name=model_name)
            # disabled → stays None
        return self._sparse_embedder

    def _sparse_embed_query(self, query: str) -> QdrantSparseVector:
        """Generate sparse query vector using the configured mode."""
        embedder = self._get_sparse_embedder()
        if embedder is None:
            return QdrantSparseVector(indices=[], values=[])

        if self._sparse_mode == "qdrant-bm25":
            sv = embedder.embed_text(query)
            return QdrantSparseVector(indices=sv.indices, values=sv.values)
        else:
            # SPLADE mode — fastembed SparseTextEmbedding
            results = list(embedder.embed([query]))
            embedding = results[0]
            return QdrantSparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )

    def search(
        self,
        query: str,
        operator: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        執行搜索

        Args:
            query: 查詢字串
            operator: hybrid | semantic | keyword | exact
            filters: 過濾條件 {"source_kind": ["code"], "category": "...", ...}
            top_n: 返回數量

        Returns:
            搜索結果列表
        """
        top_n = top_n or self.config.top_n

        if operator == "hybrid":
            if self._check_sparse_support():
                return self._hybrid_search(query, filters, top_n)
            else:
                # Fallback: no sparse vectors in collection
                return self._semantic_search(query, filters, top_n)
        elif operator == "semantic":
            return self._semantic_search(query, filters, top_n)
        elif operator == "keyword":
            if self._check_sparse_support():
                return self._keyword_search(query, filters, top_n)
            else:
                # Fallback: semantic with enhanced query
                return self._semantic_search(f"exact match: {query}", filters, top_n)
        elif operator == "exact":
            return self._exact_search(query, filters, top_n)
        else:
            return self._hybrid_search(query, filters, top_n) if self._check_sparse_support() \
                else self._semantic_search(query, filters, top_n)

    def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """語義搜索 - 使用 dense 向量相似度"""
        query_vector = self._embed_query(query)
        qdrant_filter = self._build_filter(filters)

        # Use named vector if collection has sparse support (named vectors)
        if self._check_sparse_support():
            results = self.qdrant.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                using="dense",
                limit=top_n,
                query_filter=qdrant_filter,
                with_payload=True
            )
        else:
            results = self.qdrant.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                limit=top_n,
                query_filter=qdrant_filter,
                with_payload=True
            )

        return self._format_results(results.points, "semantic")

    def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """關鍵字搜索 - 使用 BM25 sparse vector"""
        sparse_vector = self._sparse_embed_query(query)
        qdrant_filter = self._build_filter(filters)

        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=sparse_vector,
            using="sparse",
            limit=top_n,
            query_filter=qdrant_filter,
            with_payload=True
        )

        return self._format_results(results.points, "keyword")

    def _hybrid_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """混合搜索 - prefetch dense + sparse, RRF fusion"""
        dense_vector = self._embed_query(query)
        sparse_vector = self._sparse_embed_query(query)
        qdrant_filter = self._build_filter(filters)

        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_n,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_n,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_n,
            with_payload=True
        )

        return self._format_results(results.points, "hybrid")

    def _exact_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """精確搜索 - 嘗試在 payload 中匹配"""
        clean_query = query.strip('"\'')

        # Use hybrid or semantic to get candidates
        if self._check_sparse_support():
            results = self._hybrid_search(query, filters, top_n * 2)
        else:
            results = self._semantic_search(query, filters, top_n * 2)

        # Filter for exact string matches in content
        filtered = []
        for r in results:
            content = r.get("content", "") or r.get("payload", {}).get("content_preview", "")
            if clean_query.lower() in content.lower():
                r["exact_match"] = True
                filtered.append(r)

        # Supplement with non-exact results if too few
        if len(filtered) < top_n // 2:
            for r in results:
                if r not in filtered:
                    r["exact_match"] = False
                    filtered.append(r)
                    if len(filtered) >= top_n:
                        break

        return filtered[:top_n]

    def _embed_query(self, query: str) -> List[float]:
        """生成查詢向量 (uses cached embedding from batch if available)"""
        cached = getattr(self, '_cached_embedding', None)
        if cached is not None:
            return cached
        response = self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=query
        )
        return response.data[0].embedding

    def _embed_queries_batch(self, queries: List[str]) -> List[List[float]]:
        """批量生成查詢向量 (OpenAI API 支持 batch input)"""
        if not queries:
            return []
        if len(queries) == 1:
            return [self._embed_query(queries[0])]

        response = self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=queries
        )
        # API returns embeddings in order of input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Filter]:
        """構建 Qdrant 過濾器"""
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def _format_results(
        self,
        results: List,
        search_type: str
    ) -> List[Dict[str, Any]]:
        """格式化搜索結果"""
        formatted = []

        for hit in results:
            payload = hit.payload or {}

            formatted.append({
                "id": str(hit.id),
                "score": hit.score,
                "search_type": search_type,
                "path": payload.get("file_path", ""),
                "content": payload.get("content_preview", ""),
                "category": payload.get("category", ""),
                "service": payload.get("service", ""),
                "layer": payload.get("layer", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "total_chunks": payload.get("total_chunks", 1),
                "payload": payload
            })

        return formatted

    def search_by_file_path(self, file_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks by exact file_path match.

        Used by graph search enhancer to fetch content for neighbor files.
        """
        from qdrant_client.http import models as rest

        try:
            results = self.qdrant.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_path",
                            match=rest.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
            formatted = []
            for point in results[0]:
                payload = point.payload or {}
                formatted.append({
                    "id": str(point.id),
                    "path": payload.get("file_path", ""),
                    "content": payload.get("content_preview", ""),
                    "score": 0.5,  # No scoring for direct fetch
                    "score_hybrid": 0.5,
                    "payload": payload,
                })
            return formatted
        except Exception as e:
            logger.warning(f"search_by_file_path failed for {file_path}: {e}")
            return []

    def count_tokens(self, text: str) -> int:
        """計算 token 數量"""
        return len(self.encoding.encode(text))

    def get_collection_info(self) -> Dict[str, Any]:
        """獲取 collection 信息"""
        try:
            info = self.qdrant.get_collection(self.config.collection_name)
            return {
                "name": self.config.collection_name,
                "points_count": info.points_count,
                "status": str(info.status),
                "has_sparse": self._check_sparse_support(),
            }
        except Exception as e:
            return {"error": str(e)}


class HybridSearchBatch:
    """批量混合搜索"""

    def __init__(self, search_engine: HybridSearch):
        self.engine = search_engine

    def search_batch(
        self,
        queries: List[Dict[str, Any]],
        top_n_per_query: int = 50
    ) -> List[Dict[str, Any]]:
        """
        批量搜索並合併結果

        Pre-computes all dense embeddings in a single API call for efficiency.

        Args:
            queries: [{"query": "...", "operator": "...", "filters": {...}}, ...]
            top_n_per_query: 每個查詢的結果數

        Returns:
            合併去重後的結果
        """
        if not queries:
            return []

        # Pre-compute dense embeddings in batch for queries that need them
        needs_dense = [
            q.get("operator", "hybrid") in ("hybrid", "semantic", "exact", "")
            for q in queries
        ]
        dense_queries = [q.get("query", "") for q, nd in zip(queries, needs_dense) if nd]

        embeddings_map: Dict[str, List[float]] = {}
        if dense_queries:
            unique_queries = list(dict.fromkeys(dense_queries))  # dedupe preserving order
            batch_embeddings = self.engine._embed_queries_batch(unique_queries)
            embeddings_map = dict(zip(unique_queries, batch_embeddings))

        all_results = []
        seen_ids = set()

        for q in queries:
            query_text = q.get("query", "")
            operator = q.get("operator", "hybrid")

            # Inject pre-computed embedding via _cached_embedding
            cached = embeddings_map.get(query_text)
            if cached is not None:
                self.engine._cached_embedding = cached
            else:
                self.engine._cached_embedding = None

            results = self.engine.search(
                query=query_text,
                operator=operator,
                filters=q.get("filters"),
                top_n=top_n_per_query
            )

            # Clear cache
            self.engine._cached_embedding = None

            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_results.append(r)

        # 按分數排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results
