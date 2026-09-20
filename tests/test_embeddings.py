import numpy as np
import pytest

from stock_advisor.rag.embeddings import TfidfEmbeddingModel


def test_embed_before_fit_raises():
    model = TfidfEmbeddingModel()
    with pytest.raises(RuntimeError):
        model.embed(["some text"])


def test_identical_text_embeds_to_itself_with_similarity_one():
    model = TfidfEmbeddingModel()
    documents = [
        "the company faces significant litigation risk this year",
        "revenue grew steadily across all business segments",
        "the board approved a new capital expenditure plan",
    ]
    model.fit(documents)

    vectors = model.embed([documents[0], documents[0]])
    similarity = np.dot(vectors[0], vectors[1]) / (
        np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
    )
    assert similarity == pytest.approx(1.0, abs=1e-6)


def test_related_text_scores_higher_than_unrelated_text():
    model = TfidfEmbeddingModel()
    documents = [
        "the company faces significant pending litigation and regulatory risk",
        "quarterly revenue grew steadily across all business segments",
        "the cafeteria menu was updated to include more vegetarian options",
    ]
    model.fit(documents)

    query_vec = model.embed(["litigation and regulatory risk"])[0]
    doc_vecs = model.embed(documents)

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    scores = [cosine(query_vec, doc_vec) for doc_vec in doc_vecs]
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
