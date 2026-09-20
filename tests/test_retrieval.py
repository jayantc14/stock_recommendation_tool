import numpy as np
import pytest

from stock_advisor.rag.chunking import TextChunk
from stock_advisor.rag.retrieval import VectorStore


def test_query_returns_closest_vectors_ranked_by_similarity():
    chunks = [
        TextChunk(text="litigation risk chunk", chunk_index=0),
        TextChunk(text="revenue growth chunk", chunk_index=1),
        TextChunk(text="cafeteria menu chunk", chunk_index=2),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    store = VectorStore()
    store.index(chunks, vectors)

    results = store.query(np.array([0.9, 0.1, 0.0]), top_k=2)

    assert len(results) == 2
    assert results[0].chunk.text == "litigation risk chunk"
    assert results[0].score > results[1].score


def test_query_on_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.query(np.array([1.0, 0.0]), top_k=5) == []


def test_index_rejects_mismatched_lengths():
    chunks = [TextChunk(text="a", chunk_index=0)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    store = VectorStore()
    with pytest.raises(ValueError):
        store.index(chunks, vectors)


def test_query_excludes_zero_or_negative_similarity_results():
    chunks = [
        TextChunk(text="matches", chunk_index=0),
        TextChunk(text="opposite", chunk_index=1),
    ]
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    store = VectorStore()
    store.index(chunks, vectors)

    results = store.query(np.array([1.0, 0.0]), top_k=5)

    assert len(results) == 1
    assert results[0].chunk.text == "matches"
