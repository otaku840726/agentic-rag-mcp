"""
Reranker - Cross-encoder 重排序
使用 sentence-transformers 的 cross-encoder 模型
比 LLM 更快、更穩定、更便宜
"""

import os
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .provider import get_component_config, create_client

# 延遲導入以避免不必要的加載
_cross_encoder = None


def get_cross_encoder(model_name: str):
    """延遲加載 cross-encoder 模型"""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


@dataclass
class RerankerConfig:
    """Reranker 配置"""
    provider: str = "local"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_m: int = 20  # 保留的結果數量
    batch_size: int = 32


class Reranker:
    """Cross-encoder 重排序器 (支援 Local 模型與遠端 API)"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._model = None
        self._api_client = None

        # Determine if we use local model or remote API
        if self.config.provider != "local":
            self._api_client = create_client(self.config.provider)

    @property
    def model(self):
        """延遲加載模型 (僅 local)"""
        if self._model is None and self.config.provider == "local":
            self._model = get_cross_encoder(self.config.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_m: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        重排序候選結果

        Args:
            query: 原始查詢
            candidates: 候選結果列表，每個必須包含 'content' 字段
            top_m: 保留數量

        Returns:
            重排序後的結果（包含 score_rerank）
        """
        top_m = top_m or self.config.top_m

        if not candidates:
            return []

        # 擷取內容
        documents = []
        for c in candidates:
            content = c.get("content", "") or c.get("payload", {}).get("content_preview", "")
            if len(content) > 512 and self.config.provider == "local":
                content = content[:512]
            documents.append(content)

        scores = []

        # 1. 如果是 Voyage AI API
        if self.config.provider == "voyage":
            try:
                import voyageai
                client = voyageai.Client(api_key=self._api_client.api_key)
                rerank_res = client.rerank(
                    query=query,
                    documents=documents,
                    model=self.config.model_name,
                    top_k=top_m
                )
                # 初始化 scores，將所有 candidates 設為 0
                scores = [0.0] * len(candidates)
                for res in rerank_res.results:
                    scores[res.index] = res.relevance_score
            except Exception as e:
                print(f"Voyage rerank API failed: {e}. Falling back to original scores.")
                scores = [c.get("score", 0.0) for c in candidates]

        # 2. 如果是其他 OpenAI compatible 的 provider，且假設他們有 /v1/rerank 端點（如 Jina, Cohere compatible）
        elif self.config.provider != "local" and self._api_client:
            try:
                # Using standard requests as OpenAI client doesn't support /v1/rerank directly yet
                headers = {"Authorization": f"Bearer {self._api_client.api_key}"}
                payload = {
                    "model": self.config.model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_m
                }

                base_url = self._api_client.base_url
                if str(base_url).endswith("/v1") or str(base_url).endswith("/v1/"):
                    url = f"{str(base_url).rstrip('/')}/rerank"
                else:
                    url = f"{str(base_url).rstrip('/')}/v1/rerank"

                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    scores = [0.0] * len(candidates)
                    for res in results:
                        scores[res["index"]] = res["relevance_score"]
                else:
                    print(f"Rerank API failed ({resp.status_code}): {resp.text}. Falling back.")
                    scores = [c.get("score", 0.0) for c in candidates]
            except Exception as e:
                print(f"Rerank API failed: {e}. Falling back.")
                scores = [c.get("score", 0.0) for c in candidates]

        # 3. 如果是 local (sentence-transformers)
        else:
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs, batch_size=self.config.batch_size)
            scores = [float(s) for s in scores]

        # 添加 rerank 分數
        for i, score in enumerate(scores):
            candidates[i]["score_rerank"] = score
            candidates[i]["score_hybrid"] = candidates[i].get("score", 0)

        # 按 rerank 分數排序
        candidates.sort(key=lambda x: x.get("score_rerank", 0), reverse=True)

        return candidates[:top_m]


class SimpleReranker:
    """
    簡單重排序器 - 不依賴額外模型
    使用規則基礎的評分
    適合快速測試或資源受限的環境
    """

    def __init__(self, top_m: int = 20):
        self.top_m = top_m

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_m: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        使用規則基礎的重排序

        評分因素:
        - 原始分數
        - 關鍵字匹配
        - 精確匹配加成
        """
        top_m = top_m or self.top_m

        if not candidates:
            return []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for c in candidates:
            content = c.get("content", "") or c.get("payload", {}).get("content_preview", "")
            content_lower = content.lower()

            # 基礎分數
            base_score = c.get("score", 0)

            # 關鍵字匹配加成
            term_matches = sum(1 for t in query_terms if t in content_lower)
            term_bonus = term_matches * 0.1

            # 精確匹配加成
            exact_bonus = 0.3 if query_lower in content_lower else 0

            # 組合分數
            rerank_score = base_score + term_bonus + exact_bonus

            c["score_rerank"] = rerank_score
            c["score_hybrid"] = base_score

        # 排序
        candidates.sort(key=lambda x: x.get("score_rerank", 0), reverse=True)

        return candidates[:top_m]


def create_reranker(use_cross_encoder: bool = True) -> Any:
    """
    工廠函數 - 創建 reranker

    Args:
        use_cross_encoder: 是否使用 cross-encoder 模型 (這裡指進階 reranker 包含 api)

    Returns:
        Reranker instance
    """
    if use_cross_encoder:
        cfg = get_component_config("reranker")
        r_config = RerankerConfig(
            provider=cfg.provider,
            model_name=cfg.model,
        )
        # 嘗試初始化，若使用 local model 且沒裝函式庫則 fallback
        try:
            return Reranker(config=r_config)
        except ImportError:
            print("Warning: required libraries not installed, using SimpleReranker")
            return SimpleReranker()
    else:
        return SimpleReranker()
