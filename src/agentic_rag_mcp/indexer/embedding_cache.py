"""
Embedding 本地緩存模組
避免重複調用 OpenAI API，節省成本
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle


class EmbeddingCache:
    """Embedding 本地緩存，按 model identifier 隔離"""

    def __init__(self, cache_dir: str = None, identifier: str = None):
        """
        Args:
            cache_dir: 緩存根目錄路徑
            identifier: Embedding model 識別符，用作二級目錄名稱。
                        切換 model 時各自保留獨立 cache。
        """
        if cache_dir:
            base = Path(cache_dir)
        else:
            base = Path.cwd() / ".agentic-rag-cache" / "embeddings"

        # 以 identifier 作為子目錄，隔離不同 model 的 cache
        if identifier:
            self.cache_dir = base / identifier
        else:
            self.cache_dir = base / "_default"

        self.identifier = identifier or "_default"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 緩存索引文件
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

        # 統計
        self.hits = 0
        self.misses = 0

    def _load_index(self) -> Dict:
        """加載緩存索引"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"version": 1, "identifier": self.identifier, "entries": {}}

    def _save_index(self):
        """保存緩存索引"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _get_content_hash(self, content: str) -> str:
        """計算內容 hash"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _get_cache_path(self, content_hash: str) -> Path:
        """獲取緩存文件路徑"""
        # 使用 hash 前兩位作為子目錄，避免單目錄文件過多
        subdir = content_hash[:2]
        return self.cache_dir / subdir / f"{content_hash}.pkl"

    def get(self, content: str, model: str) -> Optional[List[float]]:
        """獲取緩存的 embedding

        Args:
            content: 文本內容
            model: embedding 模型名稱

        Returns:
            緩存的 embedding 向量，如果不存在則返回 None
        """
        content_hash = self._get_content_hash(content)

        # 檢查索引
        if content_hash not in self.index.get("entries", {}):
            self.misses += 1
            return None

        entry = self.index["entries"][content_hash]

        # 檢查模型是否匹配
        if entry.get("model") != model:
            self.misses += 1
            return None

        # 讀取緩存文件
        cache_path = self._get_cache_path(content_hash)
        if not cache_path.exists():
            self.misses += 1
            return None

        try:
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
            self.hits += 1
            return embedding
        except Exception as e:
            self.misses += 1
            return None

    def set(self, content: str, embedding: List[float], model: str, metadata: Dict = None):
        """保存 embedding 到緩存

        Args:
            content: 文本內容
            embedding: embedding 向量
            model: embedding 模型名稱
            metadata: 額外的 metadata
        """
        content_hash = self._get_content_hash(content)
        cache_path = self._get_cache_path(content_hash)

        # 創建子目錄
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 embedding
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        # 更新索引
        self.index["entries"][content_hash] = {
            "model": model,
            "dimension": len(embedding),
            "metadata": metadata or {}
        }
        self.index["identifier"] = self.identifier

        # 定期保存索引 (每 100 個)
        if len(self.index["entries"]) % 100 == 0:
            self._save_index()

    def get_batch(self, contents: List[str], model: str) -> Tuple[List[List[float]], List[int]]:
        """批量獲取緩存的 embeddings

        Args:
            contents: 文本列表
            model: embedding 模型名稱

        Returns:
            (cached_embeddings, missing_indices)
            - cached_embeddings: 緩存的 embeddings (None 表示未命中)
            - missing_indices: 未命中的索引列表
        """
        results = []
        missing_indices = []

        for i, content in enumerate(contents):
            embedding = self.get(content, model)
            results.append(embedding)
            if embedding is None:
                missing_indices.append(i)

        return results, missing_indices

    def set_batch(self, contents: List[str], embeddings: List[List[float]], model: str):
        """批量保存 embeddings

        Args:
            contents: 文本列表
            embeddings: embedding 向量列表
            model: embedding 模型名稱
        """
        for content, embedding in zip(contents, embeddings):
            self.set(content, embedding, model)

        # 保存索引
        self._save_index()

    def get_stats(self) -> Dict:
        """獲取緩存統計"""
        total_entries = len(self.index.get("entries", {}))
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "total_entries": total_entries,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "session_hit_rate": f"{hit_rate:.1%}",
            "cache_dir": str(self.cache_dir)
        }

    def clear(self):
        """清空當前 identifier 的緩存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {"version": 1, "identifier": self.identifier, "entries": {}}
        self._save_index()
        print(f"Cache cleared: {self.cache_dir}")

    def save(self):
        """保存索引"""
        self._save_index()


if __name__ == "__main__":
    # 測試
    cache = EmbeddingCache(identifier="test-model")

    # 測試緩存
    test_content = "OptimusPay payment processing"
    test_embedding = [0.1] * 1536

    # 保存
    cache.set(test_content, test_embedding, "test-model")

    # 讀取
    result = cache.get(test_content, "test-model")
    print(f"Cache test: {'PASS' if result is not None else 'FAIL'}")

    # 模型不匹配測試
    result2 = cache.get(test_content, "other-model")
    print(f"Model mismatch test: {'PASS' if result2 is None else 'FAIL'}")

    # 統計
    print(f"Stats: {cache.get_stats()}")
    print(f"Cache dir: {cache.cache_dir}")
