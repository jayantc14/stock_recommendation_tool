import json

from stock_advisor.rag.risk_extraction import extract_risk_flags


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None
        self.last_system = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return self.response


_DOCUMENT = """
Business Overview

The company operates in the consumer electronics sector across three
continents, with manufacturing facilities in two countries and a growing
direct-to-consumer online channel.

Contingent Liabilities and Litigation

The company is a defendant in a pending class-action lawsuit alleging
defective battery components in its flagship product line. Management
estimates a reasonably possible loss of up to fifty million dollars,
though no provision has been recorded as the outcome remains uncertain.
Separately, a regulatory inquiry by the environmental protection agency
into factory emissions remains ongoing as of the filing date.

Marketing and Distribution

The company distributes its products through a network of retail partners
and its own online storefront, with distribution centers in four regions
supporting next-day delivery in major metropolitan areas.
"""


def test_extract_risk_flags_parses_grounded_llm_response():
    canned_response = json.dumps(
        [
            {
                "risk_type": "litigation",
                "severity": "medium",
                "summary": "Pending class-action lawsuit over defective battery components.",
                "source_excerpt": "pending class-action lawsuit alleging defective battery components",
            },
            {
                "risk_type": "regulatory",
                "severity": "low",
                "summary": "Ongoing environmental regulatory inquiry into factory emissions.",
                "source_excerpt": "regulatory inquiry by the environmental protection agency",
            },
        ]
    )
    llm = _FakeLLMClient(canned_response)

    flags = extract_risk_flags(_DOCUMENT, llm, top_k=2)

    assert len(flags) == 2
    assert flags[0].risk_type == "litigation"
    assert flags[1].severity == "low"


def test_prompt_sent_to_llm_is_grounded_in_retrieved_passages_only():
    llm = _FakeLLMClient("[]")

    # small chunk_size so the litigation and distribution paragraphs land
    # in different chunks -- otherwise the whole (short) test document fits
    # in a single chunk and there's nothing for top_k to exclude
    extract_risk_flags(_DOCUMENT, llm, top_k=1, chunk_size=40, overlap=5)

    # the retrieved context should contain the litigation passage...
    assert "class-action lawsuit" in llm.last_prompt
    # ...but top_k=1 means the unrelated distribution passage should NOT
    # have been included in what the LLM was shown
    assert "distribution centers in four regions" not in llm.last_prompt


def test_malformed_json_response_returns_empty_list():
    llm = _FakeLLMClient("this is not JSON at all")
    assert extract_risk_flags(_DOCUMENT, llm) == []


def test_markdown_fenced_json_is_tolerated():
    canned_response = "```json\n" + json.dumps(
        [
            {
                "risk_type": "litigation",
                "severity": "medium",
                "summary": "Summary.",
                "source_excerpt": "excerpt",
            }
        ]
    ) + "\n```"
    llm = _FakeLLMClient(canned_response)

    flags = extract_risk_flags(_DOCUMENT, llm)

    assert len(flags) == 1
    assert flags[0].risk_type == "litigation"


def test_items_missing_required_fields_are_skipped():
    canned_response = json.dumps(
        [
            {"risk_type": "litigation", "severity": "medium"},  # missing fields
            {
                "risk_type": "regulatory",
                "severity": "low",
                "summary": "Summary.",
                "source_excerpt": "excerpt",
            },
        ]
    )
    llm = _FakeLLMClient(canned_response)

    flags = extract_risk_flags(_DOCUMENT, llm)

    assert len(flags) == 1
    assert flags[0].risk_type == "regulatory"


def test_empty_document_returns_no_flags():
    llm = _FakeLLMClient("[]")
    assert extract_risk_flags("   ", llm) == []
