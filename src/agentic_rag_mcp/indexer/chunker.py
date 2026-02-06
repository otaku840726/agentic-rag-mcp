"""
文件分塊模組
根據文件類型智能分塊
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """分塊結果"""
    content: str
    index: int
    total: int
    metadata: Dict[str, Any]


class Chunker:
    """智能分塊器"""

    def __init__(self, max_chunk_size: int = 4000, overlap: int = 200):
        """
        Args:
            max_chunk_size: 最大塊大小 (字符數)
            overlap: 塊之間的重疊 (字符數)
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_file(self, content: str, file_path: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """根據文件類型分塊

        Args:
            content: 文件內容
            file_path: 文件路徑 (用於判斷類型)
            metadata: 基礎 metadata

        Returns:
            分塊列表
        """
        metadata = metadata or {}

        # 如果內容夠小，直接返回
        if len(content) <= self.max_chunk_size:
            return [Chunk(
                content=content,
                index=1,
                total=1,
                metadata={**metadata, "chunk_index": 1, "total_chunks": 1}
            )]

        # 根據文件類型選擇分塊策略
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
        chunks = []

        # 嘗試按 class 分割
        class_pattern = r'((?:public|private|internal|protected)?\s*(?:static\s+)?(?:partial\s+)?class\s+\w+[^{]*\{)'
        parts = re.split(class_pattern, content)

        if len(parts) > 1:
            # 有多個 class
            current_chunk = parts[0]  # 開頭的 using statements 等
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    class_content = parts[i] + parts[i + 1]
                    if len(current_chunk) + len(class_content) > self.max_chunk_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk)
                        current_chunk = class_content
                    else:
                        current_chunk += class_content
            if current_chunk.strip():
                chunks.append(current_chunk)
        else:
            # 單個 class 或沒有 class，使用通用分塊
            return self._chunk_generic(content, metadata)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_python(self, content: str, metadata: Dict) -> List[Chunk]:
        """Python 代碼分塊 - 按 class/function 分割"""
        chunks = []

        # 按 class 或頂級 def 分割
        pattern = r'\n(?=class\s+\w+|def\s+\w+)'
        parts = re.split(pattern, content)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part

        if current_chunk.strip():
            chunks.append(current_chunk)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_sql(self, content: str, metadata: Dict) -> List[Chunk]:
        """SQL 分塊 - 按 CREATE/ALTER 語句分割"""
        chunks = []

        # 按主要 SQL 語句分割
        pattern = r'(?=CREATE\s+(?:TABLE|PROCEDURE|FUNCTION|VIEW|INDEX)|ALTER\s+(?:TABLE|PROCEDURE))'
        parts = re.split(pattern, content, flags=re.IGNORECASE)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part

        if current_chunk.strip():
            chunks.append(current_chunk)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_yaml(self, content: str, metadata: Dict) -> List[Chunk]:
        """YAML 分塊 - 按頂級 key 分割"""
        chunks = []

        # 按頂級 key 分割 (行首非空白字符開頭)
        pattern = r'\n(?=\S)'
        parts = re.split(pattern, content)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += "\n" + part if current_chunk else part

        if current_chunk.strip():
            chunks.append(current_chunk)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_markdown(self, content: str, metadata: Dict) -> List[Chunk]:
        """Markdown 分塊 - 按標題分割"""
        chunks = []

        # 按 ## 標題分割
        pattern = r'\n(?=##\s+)'
        parts = re.split(pattern, content)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += "\n" + part if current_chunk else part

        if current_chunk.strip():
            chunks.append(current_chunk)

        return self._finalize_chunks(chunks, metadata)

    def _chunk_json(self, content: str, metadata: Dict) -> List[Chunk]:
        """JSON 分塊 - 通用分塊 (JSON 結構難以安全分割)"""
        return self._chunk_generic(content, metadata)

    def _chunk_generic(self, content: str, metadata: Dict) -> List[Chunk]:
        """通用分塊 - 按段落或固定大小"""
        chunks = []

        # 嘗試按雙換行符分割 (段落)
        paragraphs = content.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                # 如果單個段落就超過限制，強制分割
                if len(para) > self.max_chunk_size:
                    for i in range(0, len(para), self.max_chunk_size - self.overlap):
                        chunks.append(para[i:i + self.max_chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk)

        return self._finalize_chunks(chunks, metadata)

    def _finalize_chunks(self, chunks: List[str], metadata: Dict) -> List[Chunk]:
        """將字符串列表轉換為 Chunk 對象列表"""
        total = len(chunks)
        return [
            Chunk(
                content=chunk,
                index=i + 1,
                total=total,
                metadata={**metadata, "chunk_index": i + 1, "total_chunks": total}
            )
            for i, chunk in enumerate(chunks)
        ]


if __name__ == "__main__":
    # 測試
    chunker = Chunker(max_chunk_size=500)

    # 測試 C# 分塊
    csharp_code = '''
using System;

namespace OptimusPay.Data.Entities
{
    public class Deposit
    {
        public int Id { get; set; }
        public decimal Amount { get; set; }
        public string Status { get; set; }
    }

    public class Payout
    {
        public int Id { get; set; }
        public decimal Amount { get; set; }
        public string BankCode { get; set; }
    }
}
'''
    chunks = chunker.chunk_file(csharp_code, "Deposit.cs", {"service": "Internal"})
    print(f"C# chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  Chunk {chunk.index}/{chunk.total}: {len(chunk.content)} chars")
