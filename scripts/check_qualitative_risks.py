"""Manual sanity-check CLI: extract qualitative risk flags from a local
annual-report text file using a local, free, open-source LLM via Ollama.

Setup (once, on your own machine):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.2      (or phi3, mistral, any model you prefer)
    3. Ollama serves locally on http://localhost:11434 automatically

Usage:
    python scripts/check_qualitative_risks.py annual_report.txt
    python scripts/check_qualitative_risks.py annual_report.txt --model phi3
"""

from __future__ import annotations

import argparse
import json

from stock_advisor.rag.llm_client import OllamaClient
from stock_advisor.rag.risk_extraction import extract_risk_flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_path", help="Path to a plain-text annual report / filing")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name, default llama3.2")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve, default 5")
    args = parser.parse_args()

    with open(args.report_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    llm_client = OllamaClient(model=args.model, base_url=args.base_url)
    flags = extract_risk_flags(document_text, llm_client, top_k=args.top_k)

    print(f"\nFound {len(flags)} qualitative risk flag(s):\n")
    for flag in flags:
        print(json.dumps(flag.__dict__, indent=2))
        print()


if __name__ == "__main__":
    main()
