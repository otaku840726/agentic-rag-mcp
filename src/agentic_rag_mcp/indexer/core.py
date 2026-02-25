"""
IndexerService — 通用索引服務
提供 MCP tool 方法：index_files, get_status, index_by_pattern
不綁定任何特定專案結構
"""

import hashlib
import json
import os
import re
import concurrent.futures

import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

from .embedder import Embedder
from .sparse_embedder import SparseEmbedder
from .bm25_tokenizer import Bm25Tokenizer
from .chunker import Chunker
from .qdrant_ops import QdrantOps
from .analyzer import AnalyzerFactory, save_analysis_artifact
from qdrant_client.http import models

logger = logging.getLogger(__name__)

# ── Metadata schema ────────────────────────────────────────────────

# Server 自動推斷的核心字段，client 不可覆蓋
PROTECTED_METADATA_FIELDS = {
    "category",     # 自動推斷: source-code, database, documentation, configuration, data, other
    "language",     # 自動推斷: csharp, python, java, typescript, javascript, ...
    "layer",        # 自動推斷: api, business, data, job (from directory name)
    "service",      # 自動推斷: top-level directory name
    "file_path",    # 自動設置: 相對路徑
    "file_name",    # 自動設置: 檔名
    "extension",    # 自動設置: 副檔名
    "indexed_at",   # 自動設置: 索引時間
}

# Client 推薦使用的補充標籤（非強制，但建議統一命名）
# 這些會顯示在 MCP tool description 裡引導 client 使用
RECOMMENDED_CLIENT_TAGS = {
    "jira_ticket":  "Jira 工單號, e.g. 'MNY-10137', 'INIT-2098'",
    "domain":       "業務領域, e.g. 'deposit', 'payout', 'matching', 'robot', 'settlement'",
    "priority":     "索引優先級, e.g. 'high', 'normal', 'low'",
    "tags":         "自定義標籤列表, e.g. ['callback', 'merchant', 'status-flow']",
    "description":  "補充描述, e.g. 'Merchant callback data flow entry point'",
}

# ── Metadata impact classification ────────────────────────────────────

# Metadata fields that affect chunk content (used in metadata headers)
# Changes to these fields require cache invalidation and re-embedding
CONTENT_AFFECTING_METADATA = {
    "file_path",  # Used in _file_level_header() and AST _build_header()
}

# Metadata fields that only exist in Qdrant payload (not in chunk content)
# Changes to these fields can safely reuse cached embeddings
PAYLOAD_ONLY_METADATA = {
    # Client-provided supplementary tags
    "jira_ticket", "domain", "priority", "tags", "description",
    # System-generated metadata
    "indexed_at", "chunk_index", "total_chunks", "content_preview",
    # Derived/inferred metadata (not used in chunk content)
    "file_name", "extension", "category", "language", "layer", "service",
}





# ── Filtering: whitelist approach ──────────────────────────────────
# Only files with known-valuable extensions are indexed.
# Env vars allow adding more without code changes.

# Directories always excluded (regardless of extension)
DEFAULT_EXCLUDE_DIRS = {
    "bin", "obj", "node_modules", ".git", "packages",
    "__pycache__", ".venv", "venv", "dist", "build",
    "TestResults", ".vs", ".idea",
    ".agentic-rag-cache",
    "target",           # Maven/Gradle build output
    ".gradle",          # Gradle cache
}

# Filenames / suffixes always excluded (auto-generated, lock files, etc.)
EXCLUDED_FILENAME_PATTERNS = {
    ".Designer.cs", ".g.cs", ".g.i.cs", ".AssemblyInfo.cs",   # auto-generated C#
    "GlobalUsings.g.cs",                                       # .NET 6+ auto-generated
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",       # lock files
    ".min.js", ".min.css", ".map",                             # minified / sourcemaps
    ".nupkg.metadata",                                         # NuGet metadata
}

# 副檔名 → (category, language) 映射 (also serves as the base whitelist)
EXT_CATEGORY_MAP = {
    # ── .NET / C# ──
    ".cs":         ("source-code", "csharp"),
    ".cshtml":     ("source-code", "csharp"),
    ".razor":      ("source-code", "csharp"),
    ".xaml":       ("source-code", "csharp"),       # MAUI / WPF UI definitions

    # ── JVM ──
    ".java":       ("source-code", "java"),
    ".kt":         ("source-code", "kotlin"),       # Android Robot
    ".kts":        ("source-code", "kotlin"),       # Gradle build scripts

    # ── Apple ──
    ".swift":      ("source-code", "swift"),        # iOS ListenerApp

    # ── JavaScript / TypeScript ──
    ".js":         ("source-code", "javascript"),
    ".jsx":        ("source-code", "javascript"),
    ".cjs":        ("source-code", "javascript"),   # CommonJS
    ".ts":         ("source-code", "typescript"),
    ".tsx":        ("source-code", "typescript"),

    # ── Mobile ──
    ".dart":       ("source-code", "dart"),         # Flutter ListenerApp

    # ── Systems ──
    ".py":         ("source-code", "python"),
    ".go":         ("source-code", "go"),
    ".rs":         ("source-code", "rust"),
    ".h":          ("source-code", "c"),            # OCR / DeCaptcha headers
    ".c":          ("source-code", "c"),
    ".cpp":        ("source-code", "cpp"),          # OCR / DeCaptcha
    ".cc":         ("source-code", "cpp"),

    # ── Web / Style ──
    ".css":        ("source-code", "css"),
    ".html":       ("documentation", None),

    # ── HTML-based templates (server-side + SFC) ──
    ".vue":        ("source-code", "javascript"),   # Vue SFC
    ".svelte":     ("source-code", "javascript"),   # Svelte SFC
    ".erb":        ("source-code", "ruby"),         # Ruby ERB
    ".ejs":        ("source-code", "javascript"),   # EJS
    ".jsp":        ("source-code", "java"),         # Java JSP
    ".ftl":        ("source-code", "java"),         # Freemarker
    ".twig":       ("source-code", "php"),          # Twig
    ".njk":        ("source-code", "javascript"),   # Nunjucks
    ".hbs":        ("source-code", "javascript"),   # Handlebars
    ".mustache":   ("source-code", "javascript"),   # Mustache

    # ── Database ──
    ".sql":        ("database", None),

    # ── Documentation ──
    ".md":         ("documentation", None),
    ".txt":        ("documentation", None),

    # ── Configuration ──
    ".yaml":       ("configuration", None),
    ".yml":        ("configuration", None),
    ".json":       ("data", None),
    ".xml":        ("configuration", None),
    ".config":     ("configuration", None),         # NLog, App.config, Web.config
    ".properties": ("configuration", None),         # Gradle properties
    ".proto":      ("configuration", None),
    ".graphql":    ("configuration", None),

    # ── Scripts / DevOps ──
    ".sh":         ("devops", "shell"),
    ".ps1":        ("devops", "powershell"),
    ".bat":        ("devops", "batch"),
}

# Exact filenames to index (no extension or special names)
INDEXABLE_FILENAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".dockerignore",
    "Makefile",
}

# Base whitelist = all keys in EXT_CATEGORY_MAP
INDEXABLE_EXTENSIONS = set(EXT_CATEGORY_MAP.keys())


def _load_env_set(env_key: str) -> set:
    """Load a comma-separated set from env var. e.g. '.tf,.hcl,.toml' """
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return set()
    return {v.strip() for v in raw.split(",") if v.strip()}


def get_indexable_extensions() -> set:
    """Compute the final indexable extensions set = base + env additions - env removals."""
    extra = _load_env_set("INDEX_EXTRA_EXTENSIONS")       # e.g. ".tf,.hcl,.toml"
    removed = _load_env_set("INDEX_REMOVE_EXTENSIONS")    # e.g. ".json,.xml"
    return (INDEXABLE_EXTENSIONS | extra) - removed


def get_exclude_dirs() -> set:
    """Compute the final excluded dirs set = base + env additions."""
    extra = _load_env_set("INDEX_EXTRA_EXCLUDE_DIRS")     # e.g. "tmp,logs,artifacts"
    return DEFAULT_EXCLUDE_DIRS | extra


def get_excluded_filename_patterns() -> set:
    """Compute the final excluded filename patterns = base + env additions."""
    extra = _load_env_set("INDEX_EXTRA_EXCLUDE_FILES")    # e.g. ".generated.cs,.auto.ts"
    return EXCLUDED_FILENAME_PATTERNS | extra


class IndexerService:
    """通用 Codebase Indexer Service"""

    def __init__(self):
        self.base_dir = Path.cwd()
        logger.info(f"IndexerService initialized with base_dir: {self.base_dir}")
        
        # Local JSON global state (replaces QdrantStateStore)
        self._global_state_path = Path.cwd() / ".agentic-rag-cache" / "global-state.json"
        self._global_state_path.parent.mkdir(parents=True, exist_ok=True)

        self._embedder: Optional[Embedder] = None
        self._sparse_encoder = None  # Bm25Tokenizer | SparseEmbedder | None
        self._sparse_encoder_loaded = False
        self._chunker: Optional[Chunker] = None
        self._qdrant: Optional[QdrantOps] = None
        
        # Optimization: cache collection initialization status
        self._collection_initialized = False

        # Optional: Neo4j graph store
        self._graph_store = None
        self._graph_enabled: Optional[bool] = None  # None = not yet checked

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
    
    # ------------------------------------------------------------------
    # Global state helpers (local JSON, replaces QdrantStateStore)
    # ------------------------------------------------------------------

    def _load_global_state(self) -> Dict[str, Any]:
        try:
            if self._global_state_path.exists():
                with open(self._global_state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load global state: {e}")
        return {}

    def _save_global_state(self, **kwargs):
        state = self._load_global_state()
        state.update(kwargs)
        try:
            with open(self._global_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save global state: {e}")

    @property
    def graph_store(self):
        """Lazy load optional Neo4j graph store. Returns None if disabled."""
        if self._graph_enabled is None:
            from ..provider import get_neo4j_config
            neo4j_cfg = get_neo4j_config()
            self._graph_enabled = neo4j_cfg["enabled"]
        if not self._graph_enabled:
            return None
        if self._graph_store is None:
            from ..provider import get_neo4j_config
            from .graph_store import GraphStore
            cfg = get_neo4j_config()
            project = os.getenv("GRAPH_PROJECT", "default")
            self._graph_store = GraphStore(
                uri=cfg["uri"],
                username=cfg["username"],
                password=cfg["password"],
                database=cfg["database"],
                project=project,
            )
            self._graph_store.ensure_schema()
        return self._graph_store

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_embedding_identity(self) -> str:
        """獲取 embedding 模型的唯一身份標識 (優先使用 identifier)"""
        from ..provider import get_component_config
        comp_cfg = get_component_config("embedding")
        # 優先使用 identifier，如果沒設則用 model 名稱
        return comp_cfg.identifier or comp_cfg.model

    def _get_sparse_mode(self) -> str:
        from ..provider import get_sparse_config
        return get_sparse_config()["mode"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------




                
    def _needs_recreate(self, current_identity: str) -> Optional[str]:
        """檢查是否需要重建 collection。

        Returns:
            需要重建的原因，或 None。
        """
        global_state = self._load_global_state()
        stored_identity = global_state.get("embedding_model")
        # 1) identity (identifier or model) 變更
        if stored_identity and stored_identity != current_identity:
            return f"embedding identity changed ({stored_identity} -> {current_identity})"

        # 2) sparse_mode 變更
        current_sparse_mode = self._get_sparse_mode()
        stored_sparse_mode = global_state.get("sparse_mode")
        if stored_sparse_mode and stored_sparse_mode != current_sparse_mode:
            return f"sparse_mode changed ({stored_sparse_mode} -> {current_sparse_mode})"

        # 3) collection 已存在但 dimension 不匹配（legacy state 或手動建錯）
        try:
            info = self.qdrant.get_collection_info()
            if "error" not in info and info.get("exists", True):
                cfg = info.get("config", {})
                existing_dim = (cfg.get("dense", {}) or {}).get("size")
                current_dim = self._get_dimension()
                if existing_dim and existing_dim != current_dim:
                    return (
                        f"dimension mismatch (collection={existing_dim}, "
                        f"model identity {current_identity}={current_dim})"
                    )
        except Exception:
            pass

        return None

    def _get_dimension(self) -> int:
        """獲取 embedding 向量維度，優先使用 config 中的固定值，fallback 到 API probe。"""
        from ..provider import load_config
        cfg = load_config()
        config_dim = cfg.get("embedding", {}).get("dimension")
        if config_dim:
            return int(config_dim)
        # Fallback: 通過 API 探測
        return self.embedder.get_dimension()

    def _ensure_collection(self) -> Optional[str]:
        """確保 collection 存在，並檢測 embedding model / dimension / sparse_mode 變更。

        Returns:
            若重建，回傳警告訊息；否則 None。
        """
        # Optimization: skip if already initialized
        if self._collection_initialized:
            return None

        current_identity = self._get_embedding_identity()
        current_sparse_mode = self._get_sparse_mode()
        dimension = self._get_dimension()
        reason = self._needs_recreate(current_identity)
        warning = None

        if reason:
            logger.warning("Recreating collection: %s", reason)
            self.qdrant.create_collection(
                dimension=dimension,
                recreate=True,
                sparse_mode=current_sparse_mode,
            )
            self._save_global_state(
                embedding_model=current_identity,
                sparse_mode=current_sparse_mode,
            )
            warning = f"Collection recreated ({reason}). All files will be re-indexed."

        # 確保 collection 存在（首次使用時創建）
        created = self.qdrant.create_collection(
            dimension=dimension,
            sparse_mode=current_sparse_mode,
        )
        if created:
            logger.info(f"Collection ensured: {self.qdrant.collection_name} (dim={dimension}, sparse={current_sparse_mode})")

        # 保存全局狀態（如果還沒有）
        global_state = self._load_global_state()
        if not global_state.get("embedding_model"):
            self._save_global_state(
                embedding_model=current_identity,
                sparse_mode=current_sparse_mode,
            )

        # Mark as initialized
        self._collection_initialized = True
        return warning

    # ==================================================================
    # Public MCP Tool Methods
    # ==================================================================



    def get_status(self) -> Dict[str, Any]:
        """查看索引狀態（使用 Qdrant 狀態）

        自動確保 collection 存在，避免首次調用時出現 404 錯誤。
        """
        # 確保 collection 存在（首次使用時自動創建）
        try:
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Failed to ensure collection: {e}")

        try:
            qdrant_stats = self.qdrant.get_stats()
        except Exception as e:
            qdrant_stats = {"error": str(e)}

        global_state = self._load_global_state()
        # Count unique indexed files from main collection payload
        try:
            indexed_files_count = len(self.qdrant.get_all_file_hashes())
        except Exception:
            indexed_files_count = 0

        return {
            "qdrant": qdrant_stats,
            "local_state": {
                "indexed_files": indexed_files_count,
                "embedding_model": global_state.get("embedding_model"),
                "last_index_time": global_state.get("last_index_time"),
            },
        }



    def index_analysis_artifacts(self, analysis_dir: str) -> Dict[str, Any]:
        """索引預先計算的分析結果 (JSON artifacts)

        Args:
            analysis_dir: 包含 analysis result JSON 檔案的目錄路徑
        """
        warning = self._ensure_collection()
        
        analysis_path = Path(analysis_dir)
        if not analysis_path.exists():
            return {"error": f"Directory not found: {analysis_dir}"}

        results = []
        errors = []
        total_symbols = 0
        
        # Iterate over JSON files to collect file paths
        json_files = sorted(analysis_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} analysis artifacts in {analysis_dir}")

        # First pass: collect all file paths that will be indexed
        file_paths_to_index = set()
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if "file_path" not in data or "symbols" not in data:
                    continue
                    
                original_path = Path(data["file_path"])
                try:
                    rel_path = str(original_path.relative_to(self.base_dir))
                except ValueError:
                    rel_path = str(original_path)
                    if rel_path.startswith("/"):
                        rel_path = rel_path[1:]
                
                file_paths_to_index.add(rel_path)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
        
        # Determine the common directory prefix for cleanup
        # This helps us identify which old entries to delete
        if file_paths_to_index:
            # Find common prefix (e.g., "OptimusPay.Merchant.Job/")
            common_prefix = None
            for path in file_paths_to_index:
                parts = Path(path).parts
                if parts:
                    prefix = parts[0]  # Top-level directory
                    if common_prefix is None:
                        common_prefix = prefix
                    elif common_prefix != prefix:
                        common_prefix = None  # Multiple top-level dirs
                        break
            
            # Clean up old data for this directory
            if common_prefix:
                logger.info(f"Cleaning up old analysis data for directory: {common_prefix}")
                deleted_count = self._delete_by_path_prefix(common_prefix)
                logger.info(f"Deleted {deleted_count} old entries")
            else:
                logger.warning("Multiple top-level directories detected, skipping bulk cleanup")

        # Second pass: index all artifacts
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check for required fields
                if "file_path" not in data or "symbols" not in data:
                    logger.warning(f"Skipping invalid analysis artifact: {json_file}")
                    continue
                    
                original_path = Path(data["file_path"])
                # Try to make relative to base_dir if possible, else use as is (or name)
                try:
                    rel_path = str(original_path.relative_to(self.base_dir))
                except ValueError:
                     # If absolute path is not in base_dir, use original path string
                     # For remote analysis or Docker volume mapping, this might need adjustment
                     rel_path = str(original_path)
                     # Strip leading slash if present to look cleaner
                     if rel_path.startswith("/"):
                         rel_path = rel_path[1:]

                symbols = data["symbols"]
                if not symbols:
                    continue

                # Process in batches to avoid payload size limits
                BATCH_SIZE = 50
                total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
                total_file_symbols = 0
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, len(symbols))
                    batch_symbols = symbols[start_idx:end_idx]
                    
                    # Prepare for indexing
                    texts = []
                    payloads = []
                    
                    for i, sym in enumerate(batch_symbols):
                        content = sym.get("content", "")
                        if not content:
                            continue
                            
                        texts.append(content)
                        
                        # Construct payload
                        # Merge symbol data with file metadata
                        payload = {
                            "file_path": rel_path,
                            "file_name": original_path.name,
                            "extension": original_path.suffix,
                            "chunk_index": start_idx + i,
                            "total_chunks": len(symbols),
                            "content_preview": content[:500],
                            "symbol_name": sym.get("name"),
                            "symbol_type": sym.get("node_type"),
                            "category": "source-code"
                        }
                        
                        # Add other symbol fields as metadata, excluding content
                        for k, v in sym.items():
                            if k not in ["content", "metadata_header"]:
                                 payload[k] = v
                        
                        payloads.append(payload)

                    if not texts:
                        continue

                    # Embed
                    embeddings = self.embedder.embed_batch(texts)
                    sparse_embeddings = (
                        self.sparse_encoder.embed_batch(texts)
                        if self.sparse_encoder is not None
                        else None
                    )

                    # Upsert
                    # First batch: use upsert_by_file_path to clear old data for this file
                    # Subsequent batches: use upsert to append
                    if batch_idx == 0:
                        self.qdrant.upsert_by_file_path(
                            rel_path, embeddings, payloads, sparse_vectors=sparse_embeddings
                        )
                    else:
                        self.qdrant.upsert(
                            embeddings, payloads, sparse_vectors=sparse_embeddings
                        )
                    
                    total_file_symbols += len(texts)
                
                total_symbols += total_file_symbols
                results.append({"file": rel_path, "symbols": total_file_symbols})
                logger.info(f"Indexed analysis for {rel_path}: {total_file_symbols} symbols in {total_batches} batches")

                # Push to graph store if enabled
                if self.graph_store:
                    try:
                        # Delete old graph data for this file first
                        self.graph_store.delete_by_file(rel_path, project=self.graph_store.default_project)

                        # Upsert file node
                        file_meta = {
                            "service": rel_path.split("/")[0] if "/" in rel_path else "",
                            "layer": "",
                            "category": "source-code",
                            "language": data.get("language", ""),
                        }
                        self.graph_store.upsert_file_node(rel_path, file_meta, project=self.graph_store.default_project)

                        # Upsert symbols with FQN
                        # Roslyn analyzer stores full FQN in both name and fqn fields,
                        # and puts the actual .cs file path in sym["metadata"]["file_path"]
                        # (Docker-mounted at /src/). We fix both issues here.
                        sln_dir = str(Path(rel_path).parent)  # e.g. "OptimusPay.Merchant"
                        graph_symbols = []
                        extra_rels = []
                        for sym in symbols:
                            fqn = sym.get("fqn") or sym.get("name", "unknown")
                            # Short name: last segment of FQN (e.g. "DepositService").
                            # Strip param types for Java method FQNs like "Service.doThing(Dto,int)"
                            _raw = fqn.split(".")[-1] if "." in fqn else fqn
                            short_name = _raw.split("(")[0] if "(" in _raw else _raw

                            # Resolve actual .cs file path from per-symbol metadata.
                            # Docker mounts sln_dir at /src/ inside the container.
                            sym_meta = sym.get("metadata", {})
                            docker_path = sym_meta.get("file_path", "")
                            if docker_path.startswith("/src/"):
                                actual_file_path = sln_dir + "/" + docker_path[len("/src/"):]
                            else:
                                actual_file_path = rel_path

                            graph_symbols.append({
                                "fqn": fqn,
                                "name": short_name,
                                "kind": sym.get("node_type", ""),
                                "file_path": actual_file_path,
                                "namespace": sym_meta.get("namespace", sym.get("namespace", "")),
                                "start_line": sym.get("start_line", 0),
                                "end_line": sym.get("end_line", 0),
                                "project": self.graph_store.default_project,
                            })

                            # Roslyn doesn't emit MEMBER_OF for enum members — synthesize them.
                            if sym.get("node_type") == "enum_member" and "." in fqn:
                                parent_fqn = fqn.rsplit(".", 1)[0]
                                extra_rels.append({
                                    "type": "MEMBER_OF",
                                    "source": fqn,
                                    "target": parent_fqn,
                                })

                        if graph_symbols:
                            self.graph_store.upsert_symbols(graph_symbols)

                        # Upsert relationships (+ synthesized enum_member MEMBER_OF edges)
                        graph_rels = list(data.get("relationships", []))
                        if extra_rels:
                            graph_rels.extend(extra_rels)
                        if graph_rels:
                            self.graph_store.upsert_relationships(graph_rels)

                    except Exception as e:
                        # Surface graph write failures — do NOT silently swallow.
                        # Add to errors so the caller sees the failure in the response.
                        err_msg = f"Graph store upsert failed for {rel_path}: {e}"
                        logger.error(err_msg)
                        errors.append({"file": rel_path, "error": err_msg})

            except Exception as e:
                errors.append({"file": json_file.name, "error": str(e)})
                logger.error(f"Error indexing artifact {json_file}: {e}")

        # Update last index time
        self._save_global_state(last_index_time=datetime.now().isoformat())

        resp = {
            "artifacts_processed": len(results),
            "symbols_indexed": total_symbols,
            "results": results,
            "errors": errors
        }
        if warning:
            resp["warning"] = warning
        return resp
    
    def _delete_by_path_prefix(self, prefix: str) -> int:
        """刪除指定路徑前綴的所有向量
        
        Args:
            prefix: 路徑前綴 (e.g., "OptimusPay.Merchant.Job")
            
        Returns:
            刪除的數量
        """
        try:
            # Query all entries with this prefix
            result = self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchText(text=prefix)
                        )
                    ]
                ),
                limit=10000
            )
            
            # Collect IDs to delete
            ids_to_delete = [point.id for point in result[0]]
            
            if ids_to_delete:
                # Delete by IDs
                self.qdrant.client.delete(
                    collection_name=self.qdrant.collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
                return len(ids_to_delete)
            
            return 0
        except Exception as e:
            logger.error(f"Error deleting by path prefix '{prefix}': {e}")
            return 0

    # ── Hash helpers ──────────────────────────────────────────────

    @staticmethod
    def _md5(file_path: Path) -> str:
        """Compute MD5 hex digest of a file's content."""
        h = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    @staticmethod
    def _aggregate_hash(file_hashes: Dict[str, str]) -> str:
        """Compute a single MD5 from a dict of {path: hash}, sorted by path."""
        h = hashlib.md5()
        for path in sorted(file_hashes):
            h.update(f"{path}:{file_hashes[path]}\n".encode())
        return h.hexdigest()

    @staticmethod
    def _dir_hash(dir_path: Path) -> str:
        """Compute MD5 of every file under dir_path (sorted for determinism).
        Used to detect analyzer source changes so AuraDB is re-analyzed even
        when the project's own source files haven't changed."""
        h = hashlib.md5()
        try:
            for f in sorted(dir_path.rglob("*")):
                if f.is_file():
                    try:
                        h.update(f.read_bytes())
                    except OSError:
                        pass
        except Exception:
            pass
        return h.hexdigest()

    def _is_excluded(self, file_path: Path) -> bool:
        """Whitelist-based filtering.

        A file is included if:
          - Its extension is in the indexable whitelist, OR
          - Its exact filename is in INDEXABLE_FILENAMES (e.g. Dockerfile)
        AND it is NOT in an excluded directory or matched by an excluded filename pattern.
        """
        parts = file_path.parts
        fname = file_path.name

        # 1. Excluded directories
        if any(p in get_exclude_dirs() for p in parts):
            return True

        # 2. Excluded filename patterns (suffix match, e.g. ".Designer.cs")
        for pattern in get_excluded_filename_patterns():
            if fname.endswith(pattern):
                return True

        # 2b. Content-hash named build artifacts (e.g. main.87f9f66e.css, 415.c8ae0361.chunk.js)
        #     Pattern: any segment is 6-20 hex chars (webpack/vite content hash)
        stem = file_path.stem  # filename without final extension
        if re.search(r'(?:^|\.)[a-f0-9]{6,20}(?:\.|$)', stem):
            return True

        # 3. Whitelist: extension OR exact filename
        ext = file_path.suffix.lower()
        if ext in get_indexable_extensions():
            return False
        if fname in INDEXABLE_FILENAMES:
            return False

        # Not in whitelist → exclude
        return True

    # ── Unified incremental index ──────────────────────────────────

    def index_codebase(self, directory: str) -> Dict[str, Any]:
        """Unified incremental codebase indexing with hash-based change detection.

        Runs two parallel pipelines:
          Pipeline A (Qdrant): ALL non-binary files → TreeSitter/Markdown/YAML → embed → Qdrant
          Pipeline B (AuraDB): .sln/.pom files whose constituent code files changed
                               → Roslyn/Spoon → AuraDB graph

        Only files/solutions with changed MD5 hashes are reprocessed.
        Deleted files are removed from both databases.
        """
        warning = self._ensure_collection()
        dir_path = Path(directory).resolve()

        # ── 1. Scan all indexable files & compute hashes ─────────
        logger.info(f"Scanning files in {dir_path}")
        current_files: Dict[str, str] = {}  # {rel_path: md5}

        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                rel_path = str(file_path.relative_to(self.base_dir))
            except ValueError:
                rel_path = str(file_path).lstrip("/")

            if self._is_excluded(Path(rel_path)):
                continue

            h = self._md5(file_path)
            if h:
                current_files[rel_path] = h

        logger.info(f"Found {len(current_files)} indexable files")

        # ── 2. Fetch stored hashes from both DBs ─────────────────
        try:
            qdrant_hashes = self.qdrant.get_all_file_hashes()
        except Exception as e:
            logger.warning(f"Could not fetch Qdrant hashes (treating all as new): {e}")
            qdrant_hashes = {}

        graph_hashes: Dict[str, str] = {}
        if self.graph_store:
            try:
                graph_hashes = self.graph_store.get_all_file_hashes()
            except Exception as e:
                logger.warning(f"Could not fetch graph hashes (treating all as new): {e}")

        # ── 3. Delta detection ────────────────────────────────────
        qdrant_changed: Set[str] = {
            p for p, h in current_files.items() if qdrant_hashes.get(p) != h
        }
        qdrant_deleted: Set[str] = set(qdrant_hashes) - set(current_files)

        # For AuraDB: group .cs files by nearest .sln, .java files by nearest pom.xml
        # Solution hash = aggregate MD5 of all constituent source files
        solutions_changed: Set[str] = set()
        graph_deleted: Set[str] = set()

        if self.graph_store:
            sln_files = [p for p in current_files if p.lower().endswith(".sln")]
            pom_files = [p for p in current_files
                         if p.endswith("pom.xml") or p.lower().endswith("build.gradle")]

            # Pre-compute analyzer source hashes so that changing SpoonAnalyzer.java
            # or RoslynAnalyzer source invalidates stored AuraDB hashes even when
            # the project's own source files haven't changed.
            _analyzers_root = Path(__file__).parent.parent / "analyzers"
            _csharp_analyzer_hash = self._dir_hash(_analyzers_root / "csharp")
            _java_analyzer_hash   = self._dir_hash(_analyzers_root / "java")

            for sln_path in sln_files:
                sln_dir = str(Path(sln_path).parent)
                constituent = {
                    p: h for p, h in current_files.items()
                    if p.startswith(sln_dir) and p.lower().endswith(".cs")
                }
                # Mix in analyzer version so Roslyn rule changes trigger re-analysis
                agg = self._aggregate_hash({
                    **constituent,
                    "__roslyn_analyzer__": _csharp_analyzer_hash,
                })
                if graph_hashes.get(sln_path) != agg:
                    solutions_changed.add(sln_path)

            for pom_path in pom_files:
                pom_dir = str(Path(pom_path).parent)
                constituent = {
                    p: h for p, h in current_files.items()
                    if p.startswith(pom_dir) and p.lower().endswith(".java")
                }
                # Mix in analyzer version so Spoon rule changes trigger re-analysis
                agg = self._aggregate_hash({
                    **constituent,
                    "__spoon_analyzer__": _java_analyzer_hash,
                })
                if graph_hashes.get(pom_path) != agg:
                    solutions_changed.add(pom_path)

            graph_deleted = set(graph_hashes) - set(current_files)

        logger.info(
            f"Qdrant: {len(qdrant_changed)} to index, {len(qdrant_deleted)} to delete | "
            f"AuraDB: {len(solutions_changed)} solutions to re-analyze, {len(graph_deleted)} to delete"
        )

        errors: List[Dict] = []

        # ── 4. Pipeline A — Qdrant (parallel with Pipeline B) ────
        qdrant_indexed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_qdrant = executor.submit(
                self._pipeline_qdrant, qdrant_changed, current_files, dir_path
            )
            future_graph = executor.submit(
                self._pipeline_graph, solutions_changed, current_files, dir_path
            ) if self.graph_store and solutions_changed else None

            qdrant_result = future_qdrant.result()
            qdrant_indexed = qdrant_result.get("indexed", 0)
            errors.extend(qdrant_result.get("errors", []))

            graph_solutions = 0
            if future_graph:
                graph_result = future_graph.result()
                graph_solutions = graph_result.get("solutions", 0)
                errors.extend(graph_result.get("errors", []))

        # ── 5. Deletions ──────────────────────────────────────────
        for path in qdrant_deleted:
            try:
                self.qdrant.delete_by_file_path(path)
            except Exception as e:
                logger.warning(f"Failed to delete Qdrant entry for {path}: {e}")

        if self.graph_store:
            for path in graph_deleted:
                try:
                    self.graph_store.delete_by_file(path)
                except Exception as e:
                    logger.warning(f"Failed to delete graph entry for {path}: {e}")

        self._save_global_state(last_index_time=datetime.now().isoformat())

        resp = {
            "files_scanned": len(current_files),
            "qdrant_indexed": qdrant_indexed,
            "qdrant_skipped": len(current_files) - len(qdrant_changed),
            "qdrant_deleted": len(qdrant_deleted),
            "graph_solutions_indexed": graph_solutions if self.graph_store else 0,
            "graph_deleted": len(graph_deleted) if self.graph_store else 0,
            "errors": errors,
            "errors_total": len(errors),
        }
        if warning:
            resp["warning"] = warning
        return resp

    def _pipeline_qdrant(
        self,
        changed_files: Set[str],
        current_files: Dict[str, str],
        dir_path: Path,
    ) -> Dict[str, Any]:
        """Pipeline A: index changed files into Qdrant via TreeSitter/Markdown/YAML."""
        from .markdown_analyzer import MarkdownAnalyzer
        from .yaml_analyzer import YAMLAnalyzer
        from ..provider import load_config as _load_cfg

        _cfg = _load_cfg()
        _doc_max_tokens = int(_cfg.get("embedding", {}).get("max_tokens", 2000))

        tree_sitter_analyzer = AnalyzerFactory.create_auto("", "tree-sitter")
        markdown_analyzer = MarkdownAnalyzer(max_tokens=_doc_max_tokens)
        yaml_analyzer = YAMLAnalyzer(max_tokens=_doc_max_tokens)

        indexed = 0
        errors: List[Dict] = []
        total_files = len(changed_files)
        log_interval = max(1, min(100, total_files // 10))

        logger.info(f"Pipeline A starting: {total_files} files to index")
        for file_num, rel_path in enumerate(sorted(changed_files), 1):
            if file_num % log_interval == 0 or file_num == total_files:
                logger.info(f"Pipeline A progress: {file_num}/{total_files} files processed, {indexed} chunks indexed so far")

            abs_path = dir_path / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            if not abs_path.exists():
                abs_path = self.base_dir / rel_path

            ext = Path(rel_path).suffix.lower()
            file_hash = current_files[rel_path]

            try:
                if ext in (".md", ".markdown"):
                    result = markdown_analyzer.analyze(str(abs_path))
                elif ext in (".yaml", ".yml"):
                    result = yaml_analyzer.analyze(str(abs_path))
                else:
                    result = tree_sitter_analyzer.analyze(str(abs_path))

                symbols = result.symbols
                if not symbols:
                    continue

                texts, payloads = [], []
                for i, sym in enumerate(symbols):
                    content = sym.get("content", "")
                    if not content:
                        continue
                    texts.append(content)
                    payload: Dict[str, Any] = {
                        "file_path": rel_path,
                        "file_name": Path(rel_path).name,
                        "extension": ext,
                        "chunk_index": i,
                        "total_chunks": len(symbols),
                        "content_preview": content[:500],
                        "symbol_name": sym.get("name"),
                        "symbol_type": sym.get("node_type"),
                        "category": EXT_CATEGORY_MAP.get(ext, ("other", None))[0],
                        "file_hash": file_hash,
                    }
                    for k, v in sym.items():
                        if k not in ("content", "metadata_header"):
                            payload[k] = v
                    payloads.append(payload)

                if not texts:
                    continue

                embeddings = self.embedder.embed_batch(texts)
                sparse = (
                    self.sparse_encoder.embed_batch(texts)
                    if self.sparse_encoder else None
                )
                self.qdrant.upsert_by_file_path(rel_path, embeddings, payloads, sparse_vectors=sparse)
                indexed += len(texts)

            except Exception as e:
                logger.error(f"Pipeline A error for {rel_path}: {e}")
                errors.append({"file": rel_path, "error": str(e)})

        logger.info(f"Pipeline A done: {indexed} chunks indexed from {len(changed_files)} files")
        return {"indexed": indexed, "errors": errors}

    def _pipeline_graph(
        self,
        solutions: Set[str],
        current_files: Dict[str, str],
        dir_path: Path,
    ) -> Dict[str, Any]:
        """Pipeline B: re-analyze changed solutions with Roslyn/Spoon → AuraDB."""
        from .project_detector import AnalyzerType
        import tempfile

        errors: List[Dict] = []
        processed = 0

        for sln_rel_path in sorted(solutions):
            abs_sln = self.base_dir / sln_rel_path
            if not abs_sln.exists():
                abs_sln = dir_path / sln_rel_path
            if not abs_sln.exists():
                continue

            ext = Path(sln_rel_path).suffix.lower()
            is_java = abs_sln.name in ("pom.xml",) or ext in (".gradle",)
            analyzer_type = AnalyzerType.SPOON if is_java else AnalyzerType.ROSLYN

            try:
                analyzer = AnalyzerFactory.create_auto(str(abs_sln), analyzer_type)
                result = analyzer.analyze(str(abs_sln))

                # Compute aggregate hash to store on the solution File node
                sln_dir = str(Path(sln_rel_path).parent)
                suffix = ".java" if is_java else ".cs"
                constituent = {
                    p: h for p, h in current_files.items()
                    if p.startswith(sln_dir) and p.lower().endswith(suffix)
                }
                agg_hash = self._aggregate_hash(constituent)

                # Write to AuraDB (reuse existing logic)
                self._ingest_analysis_to_graph(result, sln_rel_path, agg_hash)
                processed += 1

            except Exception as e:
                logger.error(f"Pipeline B error for {sln_rel_path}: {e}")
                errors.append({"solution": sln_rel_path, "error": str(e)})

        logger.info(f"Pipeline B done: {processed} solutions indexed")
        return {"solutions": processed, "errors": errors}

    def _ingest_analysis_to_graph(
        self, result: Any, sln_rel_path: str, agg_hash: str
    ):
        """Write Roslyn/Spoon AnalysisResult into AuraDB (reused from index_analysis_artifacts logic)."""
        from pathlib import Path as P
        data = {
            "file_path": sln_rel_path,
            "language": result.language,
            "symbols": result.symbols,
            "relationships": result.relationships,
        }

        sln_dir = str(P(sln_rel_path).parent)

        # Delete old data
        self.graph_store.delete_by_file(sln_rel_path, project=self.graph_store.default_project)

        # Upsert solution File node WITHOUT hash yet — hash is only committed after
        # all relationships are successfully written (see end of this method).
        file_meta = {
            "service": sln_rel_path.split("/")[0] if "/" in sln_rel_path else "",
            "layer": "",
            "category": "source-code",
            "language": data.get("language", ""),
        }
        self.graph_store.upsert_file_node(
            sln_rel_path, file_meta,
            project=self.graph_store.default_project,
            hash=None,  # intentionally null until write is confirmed complete
        )

        # Build symbol list
        graph_symbols = []
        extra_rels: List[Dict] = []
        for sym in result.symbols:
            fqn = sym.get("fqn") or sym.get("name", "unknown")
            # Strip param types for Java method FQNs like "Service.doThing(Dto,int)"
            _raw = fqn.split(".")[-1]
            short_name = _raw.split("(")[0] if "(" in _raw else _raw
            docker_path = sym.get("metadata", {}).get("file_path", "")
            actual_file_path = (
                sln_dir + "/" + docker_path[len("/src/"):]
                if docker_path.startswith("/src/")
                else sln_rel_path
            )
            graph_symbols.append({
                "fqn": fqn,
                "name": short_name,
                "kind": sym.get("node_type", ""),
                "file_path": actual_file_path,
                "namespace": sym.get("metadata", {}).get("namespace", ""),
                "start_line": sym.get("start_line", 0),
                "end_line": sym.get("end_line", 0),
                "project": self.graph_store.default_project,
            })
            # Synthesize enum MEMBER_OF edges
            if sym.get("node_type") == "enum_member":
                parent_fqn = ".".join(fqn.split(".")[:-1])
                if parent_fqn:
                    extra_rels.append({
                        "type": "MEMBER_OF",
                        "source": fqn,
                        "target": parent_fqn,
                        "kind": "",
                    })

        if graph_symbols:
            self.graph_store.upsert_symbols(graph_symbols)

        graph_rels = list(result.relationships)
        if extra_rels:
            graph_rels.extend(extra_rels)
        if graph_rels:
            # upsert_relationships raises RuntimeError on any batch failure —
            # if this raises, hash stays null so the next index-codebase run
            # will detect mismatch (null != agg) and retry the full analysis.
            self.graph_store.upsert_relationships(graph_rels)

        # All writes succeeded — now commit the hash so incremental indexing
        # knows this module is fully up-to-date.
        self.graph_store.upsert_file_node(
            sln_rel_path, file_meta,
            project=self.graph_store.default_project,
            hash=agg_hash,
        )

    def run_analysis(
        self,
        directory: str,
        output_dir: Optional[str] = None,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs static analysis on files in a directory with two-phase strategy.
        
        Phase 1: Analyze .NET/Java projects with specialized analyzers
        Phase 2: Analyze remaining files with tree-sitter
        
        Args:
            directory: Directory to analyze (relative to codebase root).
            output_dir: Optional directory to save JSON artifacts (default: .agentic-rag-cache/analysis/).
            file_extensions: Optional list of file extensions to limit analysis (e.g., ['.java', '.cs']).
            
        Returns:
            Dictionary with analysis results and statistics.
        """
        from .project_detector import ProjectDetector, AnalyzerType
        from pathlib import Path
        import os
        
        # Resolve directory path
        dir_path = self.base_dir / directory
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}
        
        # Default output directory: .agentic-rag-cache/analysis/{folder_name}/
        if output_dir is None:
            folder_name = Path(directory).name
            cache_dir = self.base_dir / ".agentic-rag-cache" / "analysis" / folder_name
            output_dir = str(cache_dir)
        else:
            cache_dir = Path(output_dir)
        
        # Clean up previous analysis results
        if cache_dir.exists():
            import shutil
            logger.info(f"Cleaning up previous analysis results in {cache_dir}")
            shutil.rmtree(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        detector = ProjectDetector(self.base_dir)
        
        # Track all results
        total_analyzed = 0
        total_skipped = 0
        errors = []
        results_by_analyzer = {}
        analyzed_file_paths = set()  # Track which files have been analyzed
        
        # ===== PHASE 1: Project-level analysis =====
        logger.info("Phase 1: Analyzing .NET and Java projects")
        
        # Find all .sln files first (preferred for cross-project dependencies)
        sln_files = list(dir_path.rglob("*.sln"))
        logger.info(f"Found {len(sln_files)} .sln files")
        
        # Find all .csproj files
        csproj_files = list(dir_path.rglob("*.csproj"))
        logger.info(f"Found {len(csproj_files)} .csproj files")
        
        # Track which .cs files are covered by solutions
        cs_files_in_solutions = set()
        
        # Analyze .sln files first
        if sln_files:
            image_name = os.getenv("ANALYZER_ROSLYN_IMAGE", "agentic-rag-roslyn-analyzer:latest")
            
            if detector.is_analyzer_available(AnalyzerType.ROSLYN, image_name):
                try:
                    # Use first .sln file to create analyzer
                    analyzer = AnalyzerFactory.create_auto(str(sln_files[0]), "roslyn")
                    analyzed_solutions = []
                    
                    for sln_path in sln_files:
                        rel_path = str(sln_path.relative_to(self.base_dir))
                        
                        try:
                            logger.info(f"Analyzing .NET solution: {rel_path}")
                            result = analyzer.analyze(str(sln_path))
                            save_analysis_artifact(result, output_dir)
                            analyzed_solutions.append(rel_path)
                            total_analyzed += 1
                            
                            # Mark all .cs files in this solution directory as analyzed
                            sln_dir = sln_path.parent
                            for cs_file in sln_dir.rglob("*.cs"):
                                analyzed_file_paths.add(cs_file)
                                cs_files_in_solutions.add(cs_file)
                            
                        except Exception as e:
                            errors.append({"project": rel_path, "error": str(e)})
                            logger.error(f"Error analyzing {rel_path}: {e}")
                            total_skipped += 1
                    
                    results_by_analyzer["roslyn"] = {
                        "solutions_analyzed": len(analyzed_solutions),
                        "solutions": analyzed_solutions
                    }
                    logger.info(f"Roslyn: analyzed {len(analyzed_solutions)} solutions")
                    
                except Exception as e:
                    logger.error(f"Failed to create Roslyn analyzer: {e}")
                    errors.append({"analyzer": "roslyn", "error": str(e)})
            else:
                logger.warning("Roslyn analyzer not available, skipping .NET projects")
        
        # ===== PHASE 1b: Java/Spoon project analysis =====
        pom_files = list(dir_path.rglob("pom.xml"))
        gradle_files = list(dir_path.rglob("build.gradle")) + list(dir_path.rglob("build.gradle.kts"))
        java_build_files = pom_files + gradle_files
        logger.info(f"Found {len(pom_files)} pom.xml and {len(gradle_files)} build.gradle files")

        if java_build_files:
            spoon_image = os.getenv("ANALYZER_SPOON_IMAGE", "agentic-rag-spoon-analyzer:latest")

            if detector.is_analyzer_available(AnalyzerType.SPOON, spoon_image):
                try:
                    analyzer = AnalyzerFactory.create_auto(str(java_build_files[0]), "spoon")
                    analyzed_modules = []

                    # Prefer pom.xml; for gradle-only projects use build.gradle
                    # Avoid nested poms when parent pom exists (skip sub-module poms if parent covers them)
                    # Simple heuristic: only analyze pom.xml files that have a src/main/java sibling
                    candidate_poms = [
                        p for p in pom_files
                        if (p.parent / "src" / "main" / "java").exists()
                        or (p.parent / "src").exists()
                    ]
                    # Fallback: if no candidate found, use all pom files
                    if not candidate_poms:
                        candidate_poms = pom_files or gradle_files

                    for build_file in candidate_poms:
                        try:
                            rel_path = str(build_file.relative_to(self.base_dir))
                        except ValueError:
                            rel_path = str(build_file)  # absolute path for out-of-base projects
                        module_dir = build_file.parent

                        try:
                            logger.info(f"Analyzing Java module: {rel_path}")
                            result = analyzer.analyze(str(build_file))
                            save_analysis_artifact(result, output_dir)
                            analyzed_modules.append(rel_path)
                            total_analyzed += 1

                            # Mark build file and all .java files as analyzed (prevent Phase 2 overwrite)
                            analyzed_file_paths.add(build_file)
                            for java_file in module_dir.rglob("*.java"):
                                analyzed_file_paths.add(java_file)

                        except Exception as e:
                            errors.append({"project": rel_path, "error": str(e)})
                            logger.error(f"Error analyzing Java module {rel_path}: {e}")
                            total_skipped += 1

                    results_by_analyzer["spoon"] = {
                        "modules_analyzed": len(analyzed_modules),
                        "modules": analyzed_modules
                    }
                    logger.info(f"Spoon: analyzed {len(analyzed_modules)} Java modules")

                except Exception as e:
                    logger.error(f"Failed to create Spoon analyzer: {e}")
                    errors.append({"analyzer": "spoon", "error": str(e)})
            else:
                logger.warning("Spoon analyzer not available, Java files will fall back to tree-sitter")

        # ===== PHASE 2: File-level analysis for remaining files =====
        logger.info("Phase 2: Analyzing remaining files with tree-sitter")
        
        # Collect all files not yet analyzed
        remaining_files = []
        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip if already analyzed in Phase 1
            if file_path in analyzed_file_paths:
                continue
            
            # Apply exclusion logic
            try:
                rel_path = file_path.relative_to(self.base_dir)
            except ValueError:
                rel_path = file_path  # absolute path for out-of-base projects
            if self._is_excluded(Path(rel_path)):
                continue
            
            # Filter by extension if specified
            if file_extensions:
                if file_path.suffix.lower() not in file_extensions:
                    continue
            
            # Skip project files themselves
            if file_path.suffix.lower() in ['.csproj', '.sln', '.pom', '.gradle']:
                continue
            
            remaining_files.append(file_path)
        
        logger.info(f"Found {len(remaining_files)} remaining files for tree-sitter analysis")
        
        if remaining_files:
            try:
                # Create tree-sitter analyzer (for code files)
                tree_sitter_analyzer = AnalyzerFactory.create_auto(str(remaining_files[0]), "tree-sitter")

                # Load max_tokens from config for document analyzers
                from ..provider import load_config as _load_cfg
                _cfg = _load_cfg()
                _doc_max_tokens = int(_cfg.get("embedding", {}).get("max_tokens", 2000))

                # Lazy-initialized specialized analyzers
                _markdown_analyzer = None
                _yaml_analyzer = None

                # Per-analyzer file tracking
                analyzed_by = {"tree-sitter": [], "markdown": [], "yaml": []}

                for file_path in remaining_files:
                    try:
                        rel_path = str(file_path.relative_to(self.base_dir))
                    except ValueError:
                        rel_path = str(file_path)
                    ext = file_path.suffix.lower()

                    try:
                        if ext in ('.md', '.markdown'):
                            if _markdown_analyzer is None:
                                from .markdown_analyzer import MarkdownAnalyzer
                                _markdown_analyzer = MarkdownAnalyzer(max_tokens=_doc_max_tokens)
                            result = _markdown_analyzer.analyze(str(file_path))
                            analyzer_name = "markdown"
                        elif ext in ('.yaml', '.yml'):
                            if _yaml_analyzer is None:
                                from .yaml_analyzer import YAMLAnalyzer
                                _yaml_analyzer = YAMLAnalyzer(max_tokens=_doc_max_tokens)
                            result = _yaml_analyzer.analyze(str(file_path))
                            analyzer_name = "yaml"
                        else:
                            result = tree_sitter_analyzer.analyze(str(file_path))
                            analyzer_name = "tree-sitter"

                        save_analysis_artifact(result, output_dir)
                        analyzed_by[analyzer_name].append(rel_path)
                        total_analyzed += 1
                        logger.debug(f"Analyzed {rel_path} with {analyzer_name}")

                    except Exception as e:
                        errors.append({"file": rel_path, "error": str(e)})
                        logger.error(f"Error analyzing {rel_path}: {e}")
                        total_skipped += 1

                for name, files in analyzed_by.items():
                    if files:
                        results_by_analyzer[name] = {
                            "files_analyzed": len(files),
                            "files": files
                        }
                        logger.info(f"{name}: analyzed {len(files)} files")

            except Exception as e:
                logger.error(f"Failed to create analyzers: {e}")
                errors.append({"analyzer": "phase2", "error": str(e)})
        
        logger.info(
            f"Analysis completed: {total_analyzed} items analyzed, "
            f"{total_skipped} skipped/failed"
        )
        
        return {
            "directory": directory,
            "output_dir": output_dir,
            "total_items_analyzed": total_analyzed,
            "total_items_skipped": total_skipped,
            "errors": errors,
            "results_by_analyzer": results_by_analyzer
        }


