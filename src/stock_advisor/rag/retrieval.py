from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stock_advisor.rag.chunking import TextChunk


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float


class VectorStore:
    """A minimal in-memory vector index: cosine similarity over a fixed set
    of chunk vectors. Fine for one document's worth of chunks (dozens to a
    few hundred); swap in a real vector DB (FAISS, Chroma, pgvector, ...) if
    this ever needs to scale across many documents at once.
    """

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._vectors: np.ndarray | None = None

    def index(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        self._chunks = chunks
        self._vectors = vectors

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self._vectors is None or not self._chunks:
            return []
        similarities = cosine_similarity(query_vector.reshape(1, -1), self._vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=round(float(similarities[i]), 4))
            for i in top_indices
            if similarities[i] > 0
        ]
