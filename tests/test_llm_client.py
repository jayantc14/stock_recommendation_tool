import pytest

from stock_advisor.rag.llm_client import OllamaClient


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_generate_posts_to_ollama_api_and_returns_response_text(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "here is the model's answer"})

    monkeypatch.setattr("stock_advisor.rag.llm_client.requests.post", fake_post)

    client = OllamaClient(model="llama3.2", base_url="http://localhost:11434")
    result = client.generate("summarize this", system="you are an analyst")

    assert result == "here is the model's answer"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "llama3.2"
    assert captured["json"]["prompt"] == "summarize this"
    assert captured["json"]["system"] == "you are an analyst"
    assert captured["json"]["stream"] is False


def test_generate_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        "stock_advisor.rag.llm_client.requests.post",
        lambda url, json, timeout: _FakeResponse({}, status_code=500),
    )

    client = OllamaClient()
    with pytest.raises(RuntimeError):
        client.generate("prompt")


def test_base_url_trailing_slash_is_stripped(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr("stock_advisor.rag.llm_client.requests.post", fake_post)

    client = OllamaClient(base_url="http://localhost:11434/")
    client.generate("prompt")

    assert captured["url"] == "http://localhost:11434/api/generate"
