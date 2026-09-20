from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingModel(Protocol):
    def fit(self, documents: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbeddingModel:
    """Local, download-free embedding backend.

    Fits a TF-IDF vocabulary on the document's own chunks (there's no
    pretrained model to load), then embeds text against that vocabulary.
    This is weaker at synonym/semantic matching than a neural embedding
    model (e.g. sentence-transformers), but needs no network access or
    downloaded weights -- swap in a neural `EmbeddingModel` implementation
    later without touching any other RAG module.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        self._vectorizer.fit(documents)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before embed()")
        return self._vectorizer.transform(texts).toarray()
