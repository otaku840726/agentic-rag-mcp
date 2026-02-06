"""
Sparse Embedding 模組 (SPLADE)
使用 fastembed 生成 SPLADE sparse vectors，本地運行，無 API 調用
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SparseVector:
    """Sparse vector representation"""
    indices: List[int]
    values: List[float]


class SparseEmbedder:
    """SPLADE Sparse Embedding 封裝"""

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        """
        Args:
            model_name: fastembed sparse model name (default: SPLADE++)
        """
        from fastembed import SparseTextEmbedding
        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> SparseVector:
        """生成單個文本的 sparse embedding

        Args:
            text: 輸入文本

        Returns:
            SparseVector
        """
        results = list(self.model.embed([text]))
        embedding = results[0]
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist()
        )

    def embed_batch(self, texts: List[str]) -> List[SparseVector]:
        """批量生成 sparse embeddings

        Args:
            texts: 文本列表

        Returns:
            SparseVector 列表
        """
        results = list(self.model.embed(texts))
        return [
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist()
            )
            for embedding in results
        ]
