import pytest

from stock_advisor.rag.chunking import chunk_text


def test_short_text_produces_a_single_chunk():
    chunks = chunk_text("the quick brown fox", chunk_size=200, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].text == "the quick brown fox"
    assert chunks[0].chunk_index == 0


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", chunk_size=200, overlap=40) == []


def test_long_text_is_split_with_overlap():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    # every word after the first chunk should appear in at least one chunk
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(words)

    # consecutive chunks share the overlap words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-40:] == second_words[:40]


def test_chunk_indices_are_sequential():
    words = [f"word{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_size=200, overlap=40)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
