from __future__ import annotations

import json
from dataclasses import dataclass

from stock_advisor.rag.chunking import chunk_text
from stock_advisor.rag.embeddings import EmbeddingModel, TfidfEmbeddingModel
from stock_advisor.rag.llm_client import LLMClient
from stock_advisor.rag.retrieval import VectorStore

DEFAULT_QUERY = (
    "contingent liabilities, pending litigation, related-party transactions, "
    "management changes, regulatory action, going concern risk"
)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. You will be given excerpts from "
    "a company's annual report. Identify qualitative risk flags ONLY from "
    "the provided excerpts -- do not use outside knowledge and do not "
    "invent facts not present in the text. Respond with ONLY a JSON array "
    "(no other text), where each element has the keys \"risk_type\", "
    "\"severity\" (one of \"low\", \"medium\", \"high\"), \"summary\" (one "
    "sentence), and \"source_excerpt\" (the exact quote you based this on). "
    "If no risks are found in the excerpts, return an empty array []."
)

_REQUIRED_FIELDS = ("risk_type", "severity", "summary", "source_excerpt")


@dataclass
class QualitativeRiskFlag:
    risk_type: str
    severity: str
    summary: str
    source_excerpt: str


def extract_risk_flags(
    document_text: str,
    llm_client: LLMClient,
    query: str = DEFAULT_QUERY,
    top_k: int = 5,
    chunk_size: int = 200,
    overlap: int = 40,
    embedding_model: EmbeddingModel | None = None,
) -> list[QualitativeRiskFlag]:
    """The RAG pipeline: chunk the document, retrieve the passages most
    relevant to `query`, and ask the LLM to summarize qualitative risks
    grounded ONLY in those retrieved passages -- never the whole document,
    never facts from outside it. This is the one step in the whole system
    where the LLM reasons over prose instead of computing a number, and
    grounding it in retrieval (rather than handing over the whole report)
    is what keeps it from inventing risks that aren't actually in the text.
    """
    chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return []

    model = embedding_model or TfidfEmbeddingModel()
    chunk_texts = [c.text for c in chunks]
    model.fit(chunk_texts)

    store = VectorStore()
    store.index(chunks, model.embed(chunk_texts))

    retrieved = store.query(model.embed([query])[0], top_k=top_k)
    if not retrieved:
        return []

    context = "\n\n---\n\n".join(r.chunk.text for r in retrieved)
    prompt = (
        f"Annual report excerpts:\n\n{context}\n\n"
        "Identify qualitative risk flags from ONLY the excerpts above."
    )

    raw_response = llm_client.generate(prompt, system=_SYSTEM_PROMPT)
    return _parse_risk_flags(raw_response)


def _parse_risk_flags(raw_response: str) -> list[QualitativeRiskFlag]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    flags = []
    for item in data:
        if not isinstance(item, dict) or not all(field in item for field in _REQUIRED_FIELDS):
            continue
        flags.append(QualitativeRiskFlag(**{field: item[field] for field in _REQUIRED_FIELDS}))
    return flags
