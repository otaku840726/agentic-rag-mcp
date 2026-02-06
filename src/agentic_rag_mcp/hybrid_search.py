"""
Hybrid Search - Embedding + Keyword 搜索
連接 Qdrant 向量數據庫
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from openai import OpenAI
import tiktoken


@dataclass
class SearchConfig:
    """搜索配置"""
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str
    embedding_model: str = "text-embedding-3-small"
    top_n: int = 100


class HybridSearch:
    """混合搜索引擎"""

    def __init__(self, config: Optional[SearchConfig] = None):
        # 從環境變量或配置加載
        self.config = config or SearchConfig(
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            collection_name=os.getenv("QDRANT_COLLECTION", "codebase"),
        )

        # 初始化客戶端
        self.qdrant = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key
        )
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def search(
        self,
        query: str,
        operator: str = "semantic",
        filters: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        執行搜索

        Args:
            query: 查詢字串
            operator: semantic | keyword | exact
            filters: 過濾條件 {"source_kind": ["code"], "category": "...", ...}
            top_n: 返回數量

        Returns:
            搜索結果列表
        """
        top_n = top_n or self.config.top_n

        if operator == "semantic":
            return self._semantic_search(query, filters, top_n)
        elif operator == "keyword":
            return self._keyword_search(query, filters, top_n)
        elif operator == "exact":
            return self._exact_search(query, filters, top_n)
        else:
            # 默認使用語義搜索
            return self._semantic_search(query, filters, top_n)

    def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """語義搜索 - 使用向量相似度"""
        # 生成查詢向量
        query_vector = self._embed_query(query)

        # 構建過濾器
        qdrant_filter = self._build_filter(filters)

        # 執行搜索 (使用新 API query_points)
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
        """
        關鍵字搜索
        Note: Qdrant 原生不支援全文搜索，這裡使用 payload 搜索模擬
        實際生產環境可能需要配合 Elasticsearch 或 Qdrant 的 sparse vectors
        """
        # 對於 Qdrant，我們使用 scroll + filter 來模擬關鍵字搜索
        # 或者回退到語義搜索但用關鍵字增強查詢

        # 這裡使用語義搜索作為 fallback
        # 真正的 keyword 搜索需要額外的索引支持
        enhanced_query = f"exact match: {query}"
        return self._semantic_search(enhanced_query, filters, top_n)

    def _exact_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        精確搜索 - 嘗試在 payload 中匹配
        """
        # 清理查詢
        clean_query = query.strip('"\'')

        # 先用語義搜索獲取候選
        results = self._semantic_search(query, filters, top_n * 2)

        # 然後過濾包含精確字串的結果
        filtered = []
        for r in results:
            content = r.get("content", "") or r.get("payload", {}).get("content_preview", "")
            if clean_query.lower() in content.lower():
                r["exact_match"] = True
                filtered.append(r)

        # 如果精確匹配太少，補充語義結果
        if len(filtered) < top_n // 2:
            for r in results:
                if r not in filtered:
                    r["exact_match"] = False
                    filtered.append(r)
                    if len(filtered) >= top_n:
                        break

        return filtered[:top_n]

    def _embed_query(self, query: str) -> List[float]:
        """生成查詢向量"""
        response = self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=query
        )
        return response.data[0].embedding

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Filter]:
        """構建 Qdrant 過濾器"""
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, list):
                # 多值匹配
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                )
            else:
                # 單值匹配
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

        Args:
            queries: [{"query": "...", "operator": "...", "filters": {...}}, ...]
            top_n_per_query: 每個查詢的結果數

        Returns:
            合併去重後的結果
        """
        all_results = []
        seen_ids = set()

        for q in queries:
            results = self.engine.search(
                query=q.get("query", ""),
                operator=q.get("operator", "semantic"),
                filters=q.get("filters"),
                top_n=top_n_per_query
            )

            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_results.append(r)

        # 按分數排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results
