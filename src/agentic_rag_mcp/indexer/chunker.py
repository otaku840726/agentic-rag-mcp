"""
文件分塊模組
根據文件類型智能分塊，按 token 數控制大小以匹配 embedding model 限制

Chunking strategies:
  - C# / Java: tree-sitter AST (method/class boundary), fallback to regex
  - SQL:       GO batch delimiter (T-SQL), fallback to CREATE/ALTER
  - Markdown:  ## heading boundary
  - YAML:      top-level key boundary
  - Razor:     @section / <script> / @functions blocks
  - Generic:   paragraph (\n\n) with overlap
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from dataclasses import dataclass

import tiktoken

from . import ast_chunker

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """分塊結果"""
    content: str
    index: int
    total: int
    metadata: Dict[str, Any]


class Chunker:
    """智能分塊器 — AST 優先，token 上限安全網"""

    def __init__(self, max_tokens: int = 8191, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ── Token helpers ──────────────────────────────────────────────

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _force_split_by_tokens(self, text: str) -> List[str]:
        """將超長文本按 token 數強制切割"""
        tokens = self._enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunks.append(self._enc.decode(tokens[start:end]))
            start = end - self.overlap_tokens if end < len(tokens) else end
        return chunks

    def _accumulate_parts(self, parts: List[str]) -> List[str]:
        """通用累積邏輯：將 parts 按 token 上限累積成 chunks"""
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for part in parts:
            part_tokens = self._token_len(part)

            if current_tokens + part_tokens > self.max_tokens:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                if part_tokens > self.max_tokens:
                    chunks.extend(self._force_split_by_tokens(part))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = part
                    current_tokens = part_tokens
            else:
                current_chunk += part
                current_tokens += part_tokens

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    # ── Main entry point ───────────────────────────────────────────

    def chunk_file(self, content: str, file_path: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """根據文件類型分塊"""
        metadata = metadata or {}

        # Small file: single chunk (still add metadata header for code files)
        if self._token_len(content) <= self.max_tokens:
            header = _file_level_header(file_path)
            enriched = f"{header}\n{content}" if header else content
            if self._token_len(enriched) <= self.max_tokens:
                content = enriched
            return [Chunk(
                content=content,
                index=1,
                total=1,
                metadata={**metadata, "chunk_index": 1, "total_chunks": 1}
            )]

        ext = ('.' + file_path.lower().rsplit('.', 1)[-1]) if '.' in file_path else ''

        if ext in ('.cs', '.java'):
            return self._chunk_code_ast(content, file_path, ext, metadata)
        elif ext == '.sql':
            return self._chunk_sql(content, file_path, metadata)
        elif ext == '.py':
            return self._chunk_python(content, file_path, metadata)
        elif ext in ('.yaml', '.yml'):
            return self._chunk_yaml(content, file_path, metadata)
        elif ext == '.md':
            return self._chunk_markdown(content, file_path, metadata)
        elif ext == '.cshtml':
            return self._chunk_razor(content, file_path, metadata)
        elif ext == '.json':
            return self._chunk_generic(content, file_path, metadata)
        else:
            return self._chunk_generic(content, file_path, metadata)

    # ── C# / Java: AST-based chunking ─────────────────────────────

    def _chunk_code_ast(
        self, content: str, file_path: str, ext: str, metadata: Dict
    ) -> List[Chunk]:
        """AST-based chunking for C#/Java. Falls back to regex if tree-sitter unavailable."""
        # Try AST parsing
        ast_chunks = []
        if ast_chunker.is_available(ext):
            if ext == '.cs':
                ast_chunks = ast_chunker.parse_csharp(content, file_path)
            elif ext == '.java':
                ast_chunks = ast_chunker.parse_java(content, file_path)

        if not ast_chunks:
            logger.debug("AST chunking unavailable/empty for %s, falling back to regex", file_path)
            return self._chunk_code_regex(content, file_path, ext, metadata)

        # Convert AST chunks → text parts with metadata headers
        parts: List[str] = []
        for ac in ast_chunks:
            header = ac.metadata_header
            text = f"{header}\n\n{ac.content}"
            parts.append(text)

        # Accumulate small chunks (e.g. one-liner properties) together
        # but keep large methods as individual chunks
        accumulated = self._accumulate_parts(parts)
        return self._finalize_chunks(accumulated, metadata)

    def _chunk_code_regex(
        self, content: str, file_path: str, ext: str, metadata: Dict
    ) -> List[Chunk]:
        """Regex fallback for C#/Java when tree-sitter is unavailable."""
        header = _file_level_header(file_path)

        if ext == '.cs':
            # Split by method signatures (improved regex)
            parts = re.split(
                r'\n(?=\s*(?:public|private|protected|internal)\s+'
                r'(?:static\s+)?(?:async\s+)?(?:override\s+)?'
                r'(?:virtual\s+)?(?:\w+(?:<[^>]+>)?\.?\s+)*\w+\s*\()',
                content,
            )
        elif ext == '.java':
            parts = re.split(
                r'\n(?=\s*(?:public|private|protected)\s+'
                r'(?:static\s+)?(?:final\s+)?(?:abstract\s+)?'
                r'(?:\w+(?:<[^>]+>)?\s+)+\w+\s*\()',
                content,
            )
        else:
            parts = [content]

        # Prepend file header to first part
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"

        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── SQL: GO delimiter chunking ─────────────────────────────────

    def _chunk_sql(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """SQL 分塊 — 優先用 GO 分隔符 (T-SQL)，fallback 到 CREATE/ALTER"""
        header = _file_level_header(file_path)

        # Try GO-based splitting first (T-SQL batch separator)
        go_parts = re.split(r'^\s*GO\s*$', content, flags=re.MULTILINE | re.IGNORECASE)
        go_parts = [p.strip() for p in go_parts if p.strip()]

        if len(go_parts) > 1:
            parts = []
            for p in go_parts:
                if header:
                    parts.append(f"{header}\n\n{p}")
                else:
                    parts.append(p)
            chunks = self._accumulate_parts(parts)
            return self._finalize_chunks(chunks, metadata)

        # Fallback: split by CREATE/ALTER
        parts = re.split(
            r'(?=CREATE\s+(?:TABLE|PROCEDURE|FUNCTION|VIEW|INDEX|TRIGGER)'
            r'|ALTER\s+(?:TABLE|PROCEDURE))',
            content, flags=re.IGNORECASE,
        )
        parts = [p for p in parts if p.strip()]
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"

        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── Python ─────────────────────────────────────────────────────

    def _chunk_python(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """Python 代碼分塊 — 按 class/function 分割"""
        header = _file_level_header(file_path)
        parts = re.split(r'\n(?=class\s+\w+|def\s+\w+)', content)
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── YAML ───────────────────────────────────────────────────────

    def _chunk_yaml(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """YAML 分塊 — 按頂級 key 分割"""
        header = _file_level_header(file_path)
        parts = re.split(r'\n(?=\S)', content)
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── Markdown ───────────────────────────────────────────────────

    def _chunk_markdown(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """Markdown 分塊 — 按 ## 標題分割"""
        header = _file_level_header(file_path)
        parts = re.split(r'\n(?=##\s+)', content)
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── Razor (.cshtml) ────────────────────────────────────────────

    def _chunk_razor(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """Razor view 分塊 — 按 @section / <script> / @functions 塊分割"""
        header = _file_level_header(file_path)
        parts = re.split(
            r'\n(?=@(?:section|functions|model|using)\s|<script[\s>]|</script>)',
            content,
        )
        if header and parts:
            parts[0] = f"{header}\n\n{parts[0]}"
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    # ── Generic fallback ───────────────────────────────────────────

    def _chunk_generic(self, content: str, file_path: str, metadata: Dict) -> List[Chunk]:
        """通用分塊 — 按段落分割，超長強制切割"""
        header = _file_level_header(file_path)
        paragraphs = content.split('\n\n')
        if header and paragraphs:
            paragraphs[0] = f"{header}\n\n{paragraphs[0]}"
        chunks = self._accumulate_parts(paragraphs)
        return self._finalize_chunks(chunks, metadata)

    # ── Finalize ───────────────────────────────────────────────────

    def _finalize_chunks(self, chunks: List[str], metadata: Dict) -> List[Chunk]:
        """轉換為 Chunk 列表，安全網：最終確保每個 chunk 不超限"""
        safe_chunks = []
        for chunk in chunks:
            if self._token_len(chunk) > self.max_tokens:
                safe_chunks.extend(self._force_split_by_tokens(chunk))
            else:
                safe_chunks.append(chunk)

        total = len(safe_chunks)
        return [
            Chunk(
                content=c,
                index=i + 1,
                total=total,
                metadata={**metadata, "chunk_index": i + 1, "total_chunks": total}
            )
            for i, c in enumerate(safe_chunks)
        ]


# ── Module-level helpers ───────────────────────────────────────────

def _file_level_header(file_path: str) -> str:
    """Generate a simple file-level metadata header."""
    if not file_path:
        return ""
    return f"// File: {file_path}"
