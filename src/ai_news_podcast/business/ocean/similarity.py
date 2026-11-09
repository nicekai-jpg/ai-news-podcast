"""轻量相似度索引（TF-IDF char n-gram + cosine）。

项目已有 scikit-learn，Stage 2 统一用 TF-IDF 降级路径（不依赖
sentence-transformers / torch）；字符 n-gram 对中英混排标题稳定。
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class SimilarityIndex:
    """对一组文档做一次 TF-IDF 拟合，支持批量/单条余弦相似度查询。"""

    def __init__(
        self,
        documents: list[str],
        *,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 50000,
    ) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        docs = [doc if doc and doc.strip() else " " for doc in documents]
        if docs:
            try:
                vectorizer = TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    lowercase=True,
                )
                self._matrix = vectorizer.fit_transform(docs)
                self._vectorizer = vectorizer
            except ValueError as exc:
                log.warning("TF-IDF index build failed (empty vocabulary?): %s", exc)

    @property
    def available(self) -> bool:
        return self._matrix is not None

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def matrix_similarity(self) -> np.ndarray | None:
        """全量相似度矩阵（小规模数据；调用方自行缓存）。"""
        if self._matrix is None:
            return None
        return np.asarray(cosine_similarity(self._matrix))

    def similarity(self, left: int, right: int) -> float:
        if self._matrix is None:
            return 0.0
        if left == right:
            return 1.0
        score = cosine_similarity(self._matrix[left], self._matrix[right])[0][0]
        return float(score)

    def similarities_to(self, text: str) -> np.ndarray:
        """单个文本与索引内全部文档的相似度。"""
        if self._matrix is None or self._vectorizer is None:
            return np.zeros(self.size)
        query = self._vectorizer.transform([text if text and text.strip() else " "])
        return np.asarray(cosine_similarity(query, self._matrix)[0])


def semantic_similarity(left: str, right: str) -> float:
    """两条文本的 TF-IDF 余弦相似度（单对调用，自动处理空文本）。"""
    index = SimilarityIndex([left, right])
    return index.similarity(0, 1) if index.available else 0.0
