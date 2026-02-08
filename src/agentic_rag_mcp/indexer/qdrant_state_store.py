"""
QdrantStateStore - 使用 Qdrant 存儲索引狀態，無本地文件依賴

State Collection 結構:
- Point ID: UUID (基於 file_path 的確定性哈希)
- Payload:
    - state_type: "file" | "global"
    - file_path: str (僅 file 類型)
    - hash: str (文件哈希)
    - chunks: int (chunk 數量)
    - indexed_at: str (ISO datetime)
    - embedding_model: str (僅 global 類型)
    - sparse_mode: str (僅 global 類型)
"""

import logging
import hashlib
import uuid
from typing import Dict, Optional, List
from datetime import datetime

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class QdrantStateStore:
    """
    使用 Qdrant 存儲索引狀態，替代本地 JSON 文件
    
    創建一個獨立的 collection 專門存儲狀態數據（payload-only，無向量）
    """
    
    STATE_TYPE_FILE = "file"
    STATE_TYPE_GLOBAL = "global"
    GLOBAL_STATE_ID = "00000000-0000-0000-0000-000000000001"  # 固定 UUID for global state
    
    def __init__(self, client: QdrantClient, main_collection: str):
        """
        初始化 Qdrant 狀態存儲
        
        Args:
            client: Qdrant 客戶端實例
            main_collection: 主 collection 名稱（用於生成狀態 collection 名稱）
        """
        self.client = client
        self.main_collection = main_collection
        self.state_collection = f"{main_collection}-state"
        self._ensure_state_collection()
    
    def _ensure_state_collection(self):
        """確保狀態 collection 存在，如不存在則創建"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.state_collection not in collection_names:
                logger.info(f"Creating state collection: {self.state_collection}")
                
                # 創建 payload-only collection (無向量維度)
                self.client.create_collection(
                    collection_name=self.state_collection,
                    vectors_config={},  # 空配置 = no vectors
                    on_disk_payload=True  # 將 payload 存儲在磁盤以節省內存
                )
                
                # 創建索引以加速查詢
                self.client.create_payload_index(
                    collection_name=self.state_collection,
                    field_name="state_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.state_collection,
                    field_name="file_path",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"✓ State collection created: {self.state_collection}")
        except Exception as e:
            logger.error(f"Error ensuring state collection: {e}")
            raise
    
    # === 全局狀態管理 ===
    
    def save_global_state(self, embedding_model: str, sparse_mode: str):
        """
        保存全局狀態
        
        Args:
            embedding_model: 使用的 embedding 模型名稱
            sparse_mode: sparse 模式 (bm25/fastembed/none)
        """
        payload = {
            "state_type": self.STATE_TYPE_GLOBAL,
            "embedding_model": embedding_model,
            "sparse_mode": sparse_mode,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            self.client.upsert(
                collection_name=self.state_collection,
                points=[
                    models.PointStruct(
                        id=self.GLOBAL_STATE_ID,
                        vector={},  # 空向量
                        payload=payload
                    )
                ]
            )
            logger.debug(f"Saved global state: {embedding_model}, {sparse_mode}")
        except Exception as e:
            logger.error(f"Error saving global state: {e}")
            raise
    
    def load_global_state(self) -> Optional[Dict]:
        """
        加載全局狀態
        
        Returns:
            包含全局狀態的字典，如果不存在則返回 None
        """
        try:
            points = self.client.retrieve(
                collection_name=self.state_collection,
                ids=[self.GLOBAL_STATE_ID],
                with_payload=True
            )
            if points and len(points) > 0:
                return points[0].payload
            return None
        except Exception as e:
            logger.warning(f"Error loading global state: {e}")
            return None
    
    # === 文件狀態管理 ===
    
    def save_file_state(self, file_path: str, file_hash: str, chunks: int):
        """
        保存單個文件的索引狀態
        
        Args:
            file_path: 文件路徑（相對於 codebase root）
            file_hash: 文件內容哈希值
            chunks: chunk 數量
        """
        point_id = self._generate_file_id(file_path)
        
        payload = {
            "state_type": self.STATE_TYPE_FILE,
            "file_path": file_path,
            "hash": file_hash,
            "chunks": chunks,
            "indexed_at": datetime.now().isoformat()
        }
        
        try:
            self.client.upsert(
                collection_name=self.state_collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={},  # 空向量
                        payload=payload
                    )
                ]
            )
            logger.debug(f"Saved file state: {file_path} (hash: {file_hash[:8]}...)")
        except Exception as e:
            logger.error(f"Error saving file state for {file_path}: {e}")
            raise
    
    def get_file_state(self, file_path: str) -> Optional[Dict]:
        """
        獲取文件的索引狀態
        
        Args:
            file_path: 文件路徑
            
        Returns:
            包含文件狀態的字典，如果不存在則返回 None
        """
        point_id = self._generate_file_id(file_path)
        try:
            points = self.client.retrieve(
                collection_name=self.state_collection,
                ids=[point_id],
                with_payload=True
            )
            if points and len(points) > 0:
                return points[0].payload
            return None
        except Exception as e:
            logger.debug(f"File state not found for {file_path}: {e}")
            return None
    
    def delete_file_state(self, file_path: str):
        """
        刪除文件的索引狀態
        
        Args:
            file_path: 文件路徑
        """
        point_id = self._generate_file_id(file_path)
        try:
            self.client.delete(
                collection_name=self.state_collection,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.debug(f"Deleted file state: {file_path}")
        except Exception as e:
            logger.warning(f"Error deleting file state for {file_path}: {e}")
    
    def get_all_indexed_files(self) -> List[Dict]:
        """
        獲取所有已索引文件的狀態
        
        Returns:
            文件狀態字典的列表
        """
        try:
            result, _ = self.client.scroll(
                collection_name=self.state_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="state_type",
                            match=models.MatchValue(value=self.STATE_TYPE_FILE)
                        )
                    ]
                ),
                limit=10000,
                with_payload=True
            )
            
            return [point.payload for point in result]
        except Exception as e:
            logger.error(f"Error getting all indexed files: {e}")
            return []
    
    def count_indexed_files(self) -> int:
        """
        獲取已索引文件總數
        
        Returns:
            已索引文件數量
        """
        try:
            result = self.client.count(
                collection_name=self.state_collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="state_type",
                            match=models.MatchValue(value=self.STATE_TYPE_FILE)
                        )
                    ]
                )
            )
            return result.count
        except Exception as e:
            logger.warning(f"Error getting indexed file count: {e}")
            return 0
    
    def _generate_file_id(self, file_path: str) -> str:
        """
        根據文件路徑生成確定性的 UUID
        
        Args:
            file_path: 文件路徑
            
        Returns:
            UUID 字符串
        """
        hash_obj = hashlib.md5(file_path.encode())
        # 使用 MD5 hash 的前 16 bytes 生成 UUID
        return str(uuid.UUID(bytes=hash_obj.digest()))
    
    # === 批量操作 ===
    
    def update_last_index_time(self):
        """更新最後索引時間到全局狀態"""
        try:
            # 先獲取當前全局狀態
            global_state = self.load_global_state()
            if not global_state:
                logger.warning("No global state found, cannot update last_index_time")
                return
            
            # 更新 last_index_time
            global_state["last_index_time"] = datetime.now().isoformat()
            
            self.client.upsert(
                collection_name=self.state_collection,
                points=[
                    models.PointStruct(
                        id=self.GLOBAL_STATE_ID,
                        vector={},
                        payload=global_state
                    )
                ]
            )
            logger.debug("Updated last_index_time")
        except Exception as e:
            logger.error(f"Error updating last_index_time: {e}")
    
    def bulk_update_file_states(self, file_states: List[Dict]):
        """
        批量更新文件狀態（用於遷移）
        
        Args:
            file_states: 文件狀態字典列表，每個字典包含: file_path, hash, chunks
        """
        points = []
        for state in file_states:
            point_id = self._generate_file_id(state["file_path"])
            payload = {
                "state_type": self.STATE_TYPE_FILE,
                "file_path": state["file_path"],
                "hash": state["hash"],
                "chunks": state["chunks"],
                "indexed_at": state.get("indexed_at", datetime.now().isoformat())
            }
            points.append(models.PointStruct(
                id=point_id,
                vector={},
                payload=payload
            ))
        
        # 批量插入（每次最多 100 個）
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.state_collection,
                    points=batch
                )
                logger.info(f"Bulk updated {len(batch)} file states")
            except Exception as e:
                logger.error(f"Error in bulk update batch {i//batch_size}: {e}")
                raise
    
    # === 實用工具 ===
    
    def clear_all_states(self):
        """
        清空所有狀態（危險操作，僅用於測試或重置）
        """
        try:
            self.client.delete_collection(self.state_collection)
            logger.warning(f"Deleted state collection: {self.state_collection}")
            self._ensure_state_collection()
        except Exception as e:
            logger.error(f"Error clearing states: {e}")
            raise
