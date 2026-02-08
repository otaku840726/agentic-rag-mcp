"""
IndexerService — 通用索引服務
提供 MCP tool 方法：index_files, get_status, index_by_pattern
不綁定任何特定專案結構
"""

import json
import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from .embedder import Embedder
from .sparse_embedder import SparseEmbedder
from .bm25_tokenizer import Bm25Tokenizer
from .chunker import Chunker
from .qdrant_ops import QdrantOps

logger = logging.getLogger(__name__)

# 通用排除 patterns（不需要索引的目錄/文件）
DEFAULT_EXCLUDES = {
    "bin", "obj", "node_modules", ".git", "packages",
    "__pycache__", ".venv", "venv", "dist", "build",
    "TestResults", ".vs", ".idea",
}

DEFAULT_EXCLUDE_EXTENSIONS = {
    ".dll", ".exe", ".pdb", ".pyc", ".pyo",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".rar",
}

# 副檔名 → category 映射
EXT_CATEGORY_MAP = {
    ".cs": ("source-code", "csharp"),
    ".py": ("source-code", "python"),
    ".java": ("source-code", "java"),
    ".ts": ("source-code", "typescript"),
    ".tsx": ("source-code", "typescript"),
    ".js": ("source-code", "javascript"),
    ".jsx": ("source-code", "javascript"),
    ".go": ("source-code", "go"),
    ".rs": ("source-code", "rust"),
    ".sql": ("database", None),
    ".md": ("documentation", None),
    ".yaml": ("configuration", None),
    ".yml": ("configuration", None),
    ".json": ("data", None),
    ".xml": ("configuration", None),
    ".html": ("documentation", None),
    ".css": ("source-code", "css"),
    ".dart": ("source-code", "dart"),
}


class IndexerService:
    """通用 Codebase Indexer Service"""

    def __init__(self):
        self.base_dir = Path.cwd()
        logger.info(f"IndexerService initialized with base_dir: {self.base_dir}")
        
        # 移除本地 JSON 狀態文件，改用 Qdrant
        # self.state_path = self.base_dir / ".agentic-rag-index-state.json"
        # self.state = self._load_state()

        # 初始化 Qdrant state store（延遲初始化，在 qdrant property 中創建）
        self._state_store = None

        self._embedder: Optional[Embedder] = None
        self._sparse_encoder = None  # Bm25Tokenizer | SparseEmbedder | None
        self._sparse_encoder_loaded = False
        self._chunker: Optional[Chunker] = None
        self._qdrant: Optional[QdrantOps] = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def sparse_encoder(self):
        """Return Bm25Tokenizer / SparseEmbedder / None based on sparse.mode."""
        if not self._sparse_encoder_loaded:
            from ..provider import get_sparse_config
            sparse_cfg = get_sparse_config()
            mode = sparse_cfg["mode"]
            if mode == "qdrant-bm25":
                vocab_size = int(sparse_cfg["bm25"].get("vocab_size", 30000))
                self._sparse_encoder = Bm25Tokenizer(vocab_size=vocab_size)
            elif mode == "splade":
                model = sparse_cfg["splade"].get("model", "prithivida/Splade_PP_en_v1")
                self._sparse_encoder = SparseEmbedder(model_name=model)
            else:
                self._sparse_encoder = None
            self._sparse_encoder_loaded = True
        return self._sparse_encoder

    @property
    def chunker(self) -> Chunker:
        if self._chunker is None:
            from ..provider import load_config
            cfg = load_config()
            max_tokens = int(cfg.get("embedding", {}).get("max_tokens", 8191))
            self._chunker = Chunker(max_tokens=max_tokens)
        return self._chunker

    @property
    def qdrant(self) -> QdrantOps:
        if self._qdrant is None:
            self._qdrant = QdrantOps()
        return self._qdrant
    
    @property
    def state_store(self):
        """Lazy load Qdrant state store"""
        if self._state_store is None:
            from .qdrant_state_store import QdrantStateStore
            self._state_store = QdrantStateStore(
                client=self.qdrant.client,
                main_collection=self.qdrant.collection_name
            )
        return self._state_store

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_embedding_model(self) -> str:
        from ..provider import load_config
        cfg = load_config()
        return cfg.get("embedding", {}).get("model", "text-embedding-3-small")

    def _get_sparse_mode(self) -> str:
        from ..provider import get_sparse_config
        return get_sparse_config()["mode"]

    # 移除 _load_state 和 _save_state，改用 QdrantStateStore
    # def _load_state(self) -> Dict:
    #     ...
    # def _save_state(self):
    #     ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_file_hash(self, file_path: Path) -> str:
        try:
            import xxhash
            hasher = xxhash.xxh64()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except ImportError:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

    def _should_index_file(self, file_path: Path, force: bool = False) -> bool:
        """檢查文件是否需要索引（使用 Qdrant 狀態）"""
        if force:
            return True
        rel_path = str(file_path.relative_to(self.base_dir))
        current_hash = self._get_file_hash(file_path)
        
        # 從 Qdrant 獲取文件狀態
        file_state = self.state_store.get_file_state(rel_path)
        if file_state and file_state.get("hash") == current_hash:
            return False  # 文件未變，跳過
        return True

    def _is_excluded(self, file_path: Path) -> bool:
        """通用排除邏輯"""
        parts = file_path.parts
        if any(p in DEFAULT_EXCLUDES for p in parts):
            return True
        if file_path.suffix.lower() in DEFAULT_EXCLUDE_EXTENSIONS:
            return True
        return False

    @staticmethod
    def _infer_metadata(rel_path: str) -> Dict[str, Any]:
        """從路徑純粹推斷 metadata，不依賴任何配置"""
        p = Path(rel_path)
        ext = p.suffix.lower()
        parts = p.parts

        meta: Dict[str, Any] = {}

        # 1) 副檔名 → category + language
        if ext in EXT_CATEGORY_MAP:
            category, language = EXT_CATEGORY_MAP[ext]
            meta["category"] = category
            if language:
                meta["language"] = language
        else:
            meta["category"] = "other"

        # 2) 頂層目錄 → service
        if len(parts) > 1:
            meta["service"] = parts[0]

        # 3) 目錄名 → layer hint
        dir_lower = {p.lower() for p in parts[:-1]}
        if dir_lower & {"controllers", "controller", "api"}:
            meta["layer"] = "api"
        elif dir_lower & {"business", "services", "service"}:
            meta["layer"] = "business"
        elif dir_lower & {"data", "entities", "models", "entity"}:
            meta["layer"] = "data"
        elif dir_lower & {"job", "jobs", "workers"}:
            meta["layer"] = "job"

        return meta

    def _index_file(self, file_path: Path, base_metadata: Dict) -> int:
        """索引單個文件，返回 chunk 數"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return 0

            rel_path = str(file_path.relative_to(self.base_dir))
            metadata = {
                **base_metadata,
                "file_path": rel_path,
                "file_name": file_path.name,
                "extension": file_path.suffix,
                "indexed_at": datetime.now().isoformat(),
            }

            chunks = self.chunker.chunk_file(content, rel_path, metadata)
            if not chunks:
                return 0

            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed_batch(texts)
            sparse_embeddings = (
                self.sparse_encoder.embed_batch(texts)
                if self.sparse_encoder is not None
                else None
            )

            payloads = [
                {**chunk.metadata, "content_preview": chunk.content[:500]}
                for chunk in chunks
            ]

            self.qdrant.upsert_by_file_path(
                rel_path, embeddings, payloads, sparse_vectors=sparse_embeddings
            )

            # 保存狀態到 Qdrant
            self.state_store.save_file_state(
                file_path=rel_path,
                file_hash=self._get_file_hash(file_path),
                chunks=len(chunks)
            )

            return len(chunks)

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            raise

    def _needs_recreate(self, current_model: str) -> Optional[str]:
        """檢查是否需要重建 collection。

        Returns:
            需要重建的原因，或 None。
        """
        # 從 Qdrant 獲取全局狀態
        global_state = self.state_store.load_global_state() or {}
        stored_model = global_state.get("embedding_model")

        # 1) model 名稱變更
        if stored_model and stored_model != current_model:
            return f"model changed ({stored_model} -> {current_model})"

        # 2) sparse_mode 變更
        current_sparse_mode = self._get_sparse_mode()
        stored_sparse_mode = global_state.get("sparse_mode")
        if stored_sparse_mode and stored_sparse_mode != current_sparse_mode:
            return f"sparse_mode changed ({stored_sparse_mode} -> {current_sparse_mode})"

        # 3) collection 已存在但 dimension 不匹配（legacy state 或手動建錯）
        try:
            info = self.qdrant.get_collection_info()
            if "error" not in info:
                cfg = info.get("config", {})
                existing_dim = (cfg.get("dense", {}) or {}).get("size")
                current_dim = self.embedder.get_dimension()
                if existing_dim and existing_dim != current_dim:
                    return (
                        f"dimension mismatch (collection={existing_dim}, "
                        f"model {current_model}={current_dim})"
                    )
        except Exception:
            pass

        return None

    def _ensure_collection(self) -> Optional[str]:
        """確保 collection 存在，並檢測 embedding model / dimension / sparse_mode 變更。

        Returns:
            若重建，回傳警告訊息；否則 None。
        """
        current_model = self._get_embedding_model()
        current_sparse_mode = self._get_sparse_mode()
        reason = self._needs_recreate(current_model)
        warning = None

        if reason:
            logger.warning("Recreating collection: %s", reason)
            self.qdrant.create_collection(
                dimension=self.embedder.get_dimension(),
                recreate=True,
                sparse_mode=current_sparse_mode,
            )
            # 重建後，保存新的全局狀態到 Qdrant
            self.state_store.save_global_state(
                embedding_model=current_model,
                sparse_mode=current_sparse_mode
            )
            warning = f"Collection recreated ({reason}). All files will be re-indexed."
            # 確保 collection 存在
            self.qdrant.create_collection(
                dimension=self.embedder.get_dimension(),
                sparse_mode=current_sparse_mode,
            )
            # 保存全局狀態到 Qdrant（如果還沒有）
            global_state = self.state_store.load_global_state()
            if not global_state or not global_state.get("embedding_model"):
                self.state_store.save_global_state(
                    embedding_model=current_model,
                    sparse_mode=current_sparse_mode
                )

        return warning

    # ==================================================================
    # Public MCP Tool Methods
    # ==================================================================

    def index_files(self, file_paths: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """索引指定文件

        Args:
            file_paths: 相對於 codebase root 的文件路徑列表
            metadata: 可選的額外 metadata，合併到自動推斷的 metadata
        """
        warning = self._ensure_collection()

        results = []
        errors = []
        total_chunks = 0

        for rel in file_paths:
            abs_path = self.base_dir / rel
            if not abs_path.is_file():
                errors.append({"file": rel, "error": "File not found"})
                continue

            inferred = self._infer_metadata(rel)
            if metadata:
                inferred.update(metadata)

            try:
                n_chunks = self._index_file(abs_path, inferred)
                total_chunks += n_chunks
                results.append({"file": rel, "chunks": n_chunks})
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})

        # 更新最後索引時間
        self.state_store.update_last_index_time()

        resp = {
            "files_indexed": len(results),
            "chunks_indexed": total_chunks,
            "results": results,
            "errors": errors,
        }
        if warning:
            resp["warning"] = warning
        return resp

    def get_status(self) -> Dict[str, Any]:
        """查看索引狀態（使用 Qdrant 狀態）"""
        try:
            qdrant_stats = self.qdrant.get_stats()
        except Exception as e:
            qdrant_stats = {"error": str(e)}

        # 從 Qdrant 獲取全局狀態和文件計數
        global_state = self.state_store.load_global_state() or {}
        indexed_files_count = self.state_store.count_indexed_files()
        
        return {
            "qdrant": qdrant_stats,
            "local_state": {
                "indexed_files": indexed_files_count,
                "embedding_model": global_state.get("embedding_model"),
                "last_index_time": global_state.get("last_index_time"),
            },
        }

    def index_by_pattern(
        self, pattern: str, metadata: Optional[Dict] = None, force: bool = False
    ) -> Dict[str, Any]:
        """按 glob pattern 批量索引

        Args:
            pattern: glob pattern（相對於 codebase root），例如 "knowledge/**/*.yaml"
            metadata: 可選的額外 metadata，合併到自動推斷的 metadata
            force: 強制重新索引
        """
        warning = self._ensure_collection()

        matched = sorted(self.base_dir.glob(pattern))

        files_indexed = 0
        chunks_indexed = 0
        files_skipped = 0
        errors = []

        for file_path in matched:
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(self.base_dir))
            if self._is_excluded(Path(rel_path)):
                continue

            if not self._should_index_file(file_path, force):
                files_skipped += 1
                continue

            inferred = self._infer_metadata(rel_path)
            if metadata:
                inferred.update(metadata)

            try:
                n = self._index_file(file_path, inferred)
                if n > 0:
                    files_indexed += 1
                    chunks_indexed += n
            except Exception as e:
                errors.append({"file": rel_path, "error": str(e)})

        # 更新最後索引時間
        self.state_store.update_last_index_time()

        resp = {
            "pattern": pattern,
            "files_found": len(matched),
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "files_skipped": files_skipped,
            "errors": errors,
        }
        if warning:
            resp["warning"] = warning
        return resp
