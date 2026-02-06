"""
文件分塊模組
根據文件類型智能分塊，按 token 數控制大小以匹配 embedding model 限制
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

import tiktoken


@dataclass
class Chunk:
    """分塊結果"""
    content: str
    index: int
    total: int
    metadata: Dict[str, Any]


class Chunker:
    """智能分塊器 — 按 token 數切割"""

    def __init__(self, max_tokens: int = 8191, overlap_tokens: int = 50):
        """
        Args:
            max_tokens: 每個 chunk 的最大 token 數（應 <= embedding model 限制）
            overlap_tokens: chunk 之間的重疊 token 數
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

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

    def chunk_file(self, content: str, file_path: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """根據文件類型分塊"""
        metadata = metadata or {}

        if self._token_len(content) <= self.max_tokens:
            return [Chunk(
                content=content,
                index=1,
                total=1,
                metadata={**metadata, "chunk_index": 1, "total_chunks": 1}
            )]

        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''

        if ext == 'cs':
            return self._chunk_csharp(content, metadata)
        elif ext == 'py':
            return self._chunk_python(content, metadata)
        elif ext == 'sql':
            return self._chunk_sql(content, metadata)
        elif ext in ('yaml', 'yml'):
            return self._chunk_yaml(content, metadata)
        elif ext == 'md':
            return self._chunk_markdown(content, metadata)
        elif ext == 'json':
            return self._chunk_json(content, metadata)
        else:
            return self._chunk_generic(content, metadata)

    def _chunk_csharp(self, content: str, metadata: Dict) -> List[Chunk]:
        """C# 代碼分塊 - 按 class/method 分割"""
        class_pattern = r'((?:public|private|internal|protected)?\s*(?:static\s+)?(?:partial\s+)?class\s+\w+[^{]*\{)'
        parts = re.split(class_pattern, content)

        if len(parts) > 1:
            merged = [parts[0]]
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    merged.append(parts[i] + parts[i + 1])
                else:
                    merged.append(parts[i])
            chunks = self._accumulate_parts(merged)
        else:
            return self._chunk_generic(content, metadata)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_python(self, content: str, metadata: Dict) -> List[Chunk]:
        """Python 代碼分塊 - 按 class/function 分割"""
        parts = re.split(r'\n(?=class\s+\w+|def\s+\w+)', content)
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    def _chunk_sql(self, content: str, metadata: Dict) -> List[Chunk]:
        """SQL 分塊 - 按 CREATE/ALTER 語句分割"""
        parts = re.split(
            r'(?=CREATE\s+(?:TABLE|PROCEDURE|FUNCTION|VIEW|INDEX)|ALTER\s+(?:TABLE|PROCEDURE))',
            content, flags=re.IGNORECASE,
        )
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    def _chunk_yaml(self, content: str, metadata: Dict) -> List[Chunk]:
        """YAML 分塊 - 按頂級 key 分割"""
        parts = re.split(r'\n(?=\S)', content)
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    def _chunk_markdown(self, content: str, metadata: Dict) -> List[Chunk]:
        """Markdown 分塊 - 按標題分割"""
        parts = re.split(r'\n(?=##\s+)', content)
        chunks = self._accumulate_parts(parts)
        return self._finalize_chunks(chunks, metadata)

    def _chunk_json(self, content: str, metadata: Dict) -> List[Chunk]:
        return self._chunk_generic(content, metadata)

    def _chunk_generic(self, content: str, metadata: Dict) -> List[Chunk]:
        """通用分塊 - 按段落分割，超長強制切割"""
        paragraphs = content.split('\n\n')
        chunks = self._accumulate_parts(paragraphs)
        return self._finalize_chunks(chunks, metadata)

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
