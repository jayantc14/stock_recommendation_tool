from __future__ import annotations

from typing import Protocol

import requests


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class OllamaClient:
    """Calls a local Ollama server (https://ollama.com) running a free,
    open-source model -- e.g. "llama3.2", "phi3", "mistral".

    This class only speaks HTTP to Ollama; it doesn't install Ollama, start
    its server, or pull model weights. On your own machine:

        ollama pull llama3.2
        ollama serve                 # usually already running as a service

    To use a hosted/paid model instead later (e.g. Claude), implement the
    same `LLMClient.generate` method against that provider's SDK -- nothing
    else in `stock_advisor.rag` needs to change.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"]
