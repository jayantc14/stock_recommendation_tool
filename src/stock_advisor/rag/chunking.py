from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[TextChunk]:
    """Split `text` into overlapping chunks of `chunk_size` words.

    Word-based rather than character-based, so a chunk boundary never lands
    mid-word. `overlap` words are repeated at the start of each chunk after
    the first, so a fact sitting right on a boundary is still whole in at
    least one chunk.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(TextChunk(text=" ".join(window), chunk_index=index))
        if start + chunk_size >= len(words):
            break
        start += step
        index += 1
    return chunks
