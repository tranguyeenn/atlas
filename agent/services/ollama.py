from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "prompt": text}
        response = await asyncio.to_thread(self._post_json, "/api/embeddings", payload)
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            response = await asyncio.to_thread(
                self._post_json,
                "/api/embed",
                {"model": self.embedding_model, "input": text},
            )
            embeddings = response.get("embeddings")
            if (
                not isinstance(embeddings, list)
                or not embeddings
                or not isinstance(embeddings[0], list)
            ):
                raise OllamaError("Ollama embedding response did not include an embedding.")
            embedding = embeddings[0]
        return [float(value) for value in embedding]

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {"model": self.chat_model, "messages": messages, "stream": False}
        response = await asyncio.to_thread(self._post_json, "/api/chat", payload)
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama chat response did not include message content.")
        return message["content"].strip()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}: {exc.reason}") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise OllamaError("Ollama returned an unexpected response shape.")
        return decoded
