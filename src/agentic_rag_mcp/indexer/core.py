"""
IndexerService — 通用索引服務
提供 MCP tool 方法：index_files, get_status, index_by_pattern
不綁定任何特定專案結構
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from .embedder import Embedder
from .sparse_embedder import SparseEmbedder
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
        self._pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.base_dir = self._pkg_dir.parent
        self.state_path = self._pkg_dir / "index_state.json"

        self.state = self._load_state()

        self._embedder: Optional[Embedder] = None
        self._sparse_embedder: Optional[SparseEmbedder] = None
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
    def sparse_embedder(self) -> SparseEmbedder:
        if self._sparse_embedder is None:
            self._sparse_embedder = SparseEmbedder()
        return self._sparse_embedder

    @property
    def chunker(self) -> Chunker:
        if self._chunker is None:
            self._chunker = Chunker(max_chunk_size=4000, overlap=200)
        return self._chunker

    @property
    def qdrant(self) -> QdrantOps:
        if self._qdrant is None:
            self._qdrant = QdrantOps()
        return self._qdrant

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": {}, "last_full_index": None, "last_incremental_index": None}

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False, default=str)

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
        if force:
            return True
        rel_path = str(file_path.relative_to(self.base_dir))
        current_hash = self._get_file_hash(file_path)
        if rel_path in self.state.get("files", {}):
            if self.state["files"][rel_path].get("hash") == current_hash:
                return False
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
            sparse_embeddings = self.sparse_embedder.embed_batch(texts)

            payloads = [
                {**chunk.metadata, "content_preview": chunk.content[:500]}
                for chunk in chunks
            ]

            self.qdrant.upsert_by_file_path(
                rel_path, embeddings, payloads, sparse_vectors=sparse_embeddings
            )

            self.state["files"][rel_path] = {
                "hash": self._get_file_hash(file_path),
                "chunks": len(chunks),
                "indexed_at": datetime.now().isoformat(),
            }

            return len(chunks)

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            raise

    def _ensure_collection(self):
        self.qdrant.create_collection(dimension=self.embedder.get_dimension())

    # ==================================================================
    # Public MCP Tool Methods
    # ==================================================================

    def index_files(self, file_paths: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """索引指定文件

        Args:
            file_paths: 相對於 codebase root 的文件路徑列表
            metadata: 可選的額外 metadata，合併到自動推斷的 metadata
        """
        self._ensure_collection()

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

        self._save_state()

        return {
            "files_indexed": len(results),
            "chunks_indexed": total_chunks,
            "results": results,
            "errors": errors,
        }

    def get_status(self) -> Dict[str, Any]:
        """查看索引狀態"""
        try:
            qdrant_stats = self.qdrant.get_stats()
        except Exception as e:
            qdrant_stats = {"error": str(e)}

        files_state = self.state.get("files", {})
        return {
            "qdrant": qdrant_stats,
            "local_state": {
                "indexed_files": len(files_state),
                "last_full_index": self.state.get("last_full_index"),
                "last_incremental_index": self.state.get("last_incremental_index"),
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
        self._ensure_collection()

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

        self.state["last_incremental_index"] = datetime.now().isoformat()
        self._save_state()

        return {
            "pattern": pattern,
            "files_found": len(matched),
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "files_skipped": files_skipped,
            "errors": errors,
        }
