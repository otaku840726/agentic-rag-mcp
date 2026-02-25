"""
Qdrant 向量數據庫操作模組
支持完整 CRUD 操作
支持 named vectors (dense + sparse) for hybrid search
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    Modifier,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    UpdateStatus,
    PayloadSchemaType,
    SparseVector as QdrantSparseVector,
    NamedVector,
    NamedSparseVector,
)


class QdrantOps:
    """Qdrant 向量數據庫操作封裝"""

    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None
    ):
        """
        Args:
            url: Qdrant URL (默認從環境變量)
            api_key: Qdrant API Key (可選，用於 Qdrant Cloud)
            collection_name: Collection 名稱
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "agentic-rag-codebase")

        # 初始化客戶端
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)

    def create_collection(
        self,
        dimension: int = 1536,
        recreate: bool = False,
        sparse_mode: str = "qdrant-bm25",
    ) -> bool:
        """創建 Collection (named vectors: dense + optional sparse)

        Args:
            dimension: Dense 向量維度
            recreate: 是否重新創建 (刪除現有)
            sparse_mode: "qdrant-bm25" | "splade" | "disabled"

        Returns:
            是否成功
        """
        # 檢查是否存在
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            if recreate:
                self.client.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            else:
                print(f"Collection already exists: {self.collection_name}")
                return True

        # Build sparse_vectors_config based on mode
        sparse_vectors_config = None
        if sparse_mode == "qdrant-bm25":
            sparse_vectors_config = {
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            }
        elif sparse_mode == "splade":
            sparse_vectors_config = {
                "sparse": SparseVectorParams()
            }
        # disabled → no sparse_vectors_config

        # 創建新 collection with named vectors
        create_kwargs = {
            "collection_name": self.collection_name,
            "vectors_config": {
                "dense": VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            },
        }
        if sparse_vectors_config is not None:
            create_kwargs["sparse_vectors_config"] = sparse_vectors_config

        self.client.create_collection(**create_kwargs)

        sparse_label = sparse_mode if sparse_mode != "disabled" else "none"
        print(f"Created collection: {self.collection_name} (dense dim={dimension}, sparse={sparse_label})")

        # 創建 payload indexes 用於過濾
        self._create_payload_indexes()

        return True

    def _create_payload_indexes(self):
        """創建 payload indexes 以支持過濾"""
        index_fields = ["file_path", "category", "type", "service"]
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"  Created index: {field}")
            except Exception as e:
                # Index might already exist
                pass

    def delete_collection(self) -> bool:
        """刪除 Collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def upsert(
        self,
        dense_vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[str] = None,
        sparse_vectors: List[Any] = None
    ) -> bool:
        """插入或更新向量 (支持 dense + sparse named vectors)

        Args:
            dense_vectors: Dense 向量列表
            payloads: Payload (metadata) 列表
            ids: 可選的 ID 列表，如果不提供則自動生成
            sparse_vectors: 可選的 sparse 向量列表 (SparseVector objects with .indices and .values)

        Returns:
            是否成功
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in dense_vectors]

        points = []
        for i, (id_, dense_vec, payload) in enumerate(zip(ids, dense_vectors, payloads)):
            vector = {"dense": dense_vec}
            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                vector["sparse"] = QdrantSparseVector(
                    indices=sv.indices,
                    values=sv.values
                )
            points.append(PointStruct(id=id_, vector=vector, payload=payload))

        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return result.status == UpdateStatus.COMPLETED

    def upsert_by_file_path(
        self,
        file_path: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        sparse_vectors: List[Any] = None
    ) -> bool:
        """根據文件路徑插入/更新 (先刪除同路徑的舊數據)

        Args:
            file_path: 文件路徑
            vectors: Dense 向量列表
            payloads: Payload 列表
            sparse_vectors: 可選的 sparse 向量列表

        Returns:
            是否成功
        """
        # 先刪除該文件的所有舊向量
        self.delete_by_file_path(file_path)

        # 確保所有 payload 都包含 file_path
        for payload in payloads:
            payload["file_path"] = file_path

        # 插入新向量
        return self.upsert(vectors, payloads, sparse_vectors=sparse_vectors)

    def delete_by_file_path(self, file_path: str) -> int:
        """刪除指定文件路徑的所有向量

        Args:
            file_path: 文件路徑

        Returns:
            刪除的數量
        """
        try:
            # 先查詢有多少
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=10000
            )
            count = len(result[0])

            if count > 0:
                # 刪除
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="file_path",
                                    match=MatchValue(value=file_path)
                                )
                            ]
                        )
                    )
                )

            return count
        except Exception as e:
            print(f"Error deleting by file_path: {e}")
            return 0

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """語義搜索 (using named dense vector)

        Args:
            query_vector: 查詢向量
            limit: 返回數量限制
            filter_conditions: 過濾條件 {"key": "value"}

        Returns:
            搜索結果列表
        """
        # 構建過濾器
        qdrant_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            qdrant_filter = Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results.points
        ]

    def collection_exists(self) -> bool:
        """檢查 Collection 是否存在"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """獲取 Collection 信息"""
        try:
            # 先檢查 collection 是否存在，避免 404 error
            if not self.collection_exists():
                return {
                    "name": self.collection_name,
                    "exists": False,
                    "points_count": 0,
                    "status": "not_created",
                    "message": "Collection does not exist yet. Run analyze-codebase then index-codebase to create it."
                }

            info = self.client.get_collection(self.collection_name)
            vectors_config = info.config.params.vectors

            # Handle named vectors (dict) or unnamed vector (VectorParams)
            if isinstance(vectors_config, dict):
                dense_config = vectors_config.get("dense")
                config_info = {
                    "dense": {
                        "size": dense_config.size if dense_config else None,
                        "distance": str(dense_config.distance) if dense_config else None
                    },
                    "has_sparse": bool(info.config.params.sparse_vectors)
                }
            else:
                config_info = {
                    "size": vectors_config.size,
                    "distance": str(vectors_config.distance)
                }

            return {
                "name": self.collection_name,
                "exists": True,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                "points_count": info.points_count,
                "status": info.status,
                "config": config_info
            }
        except Exception as e:
            return {"error": str(e)}

    def list_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """按類別列出索引內容

        Args:
            category: 類別名稱
            limit: 返回數量限制

        Returns:
            結果列表
        """
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [{"id": str(p.id), "payload": p.payload} for p in result[0]]

    def get_all_file_hashes(self) -> Dict[str, str]:
        """Scroll entire collection and return {file_path: file_hash} for incremental indexing.

        Only returns one hash per file_path (the first seen, duplicates are ignored —
        any re-index will overwrite them).
        """
        result: Dict[str, str] = {}
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["file_path", "file_hash"],
                with_vectors=False,
            )
            for point in batch:
                fp = point.payload.get("file_path")
                fh = point.payload.get("file_hash")
                if fp and fh and fp not in result:
                    result[fp] = fh
            if next_offset is None:
                break
            offset = next_offset
        return result

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        info = self.get_collection_info()
        if "error" in info:
            return info

        # Collection 不存在時，直接返回 info（包含 exists=False 和 message）
        if not info.get("exists", True):
            return {**info, "by_category": {}}

        # 按類別統計
        categories = ["source-code", "knowledge-base", "documentation", "database-schema", "jira", "configuration"]
        stats_by_category = {}

        for cat in categories:
            try:
                result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="category",
                                match=MatchValue(value=cat)
                            )
                        ]
                    )
                )
                stats_by_category[cat] = result.count
            except:
                stats_by_category[cat] = 0

        return {
            **info,
            "by_category": stats_by_category
        }


if __name__ == "__main__":
    # 測試
    from dotenv import load_dotenv
    load_dotenv()

    ops = QdrantOps()

    # 測試連接
    print("Collection info:", ops.get_collection_info())

    # 測試統計
    print("\nStats:", ops.get_stats())
