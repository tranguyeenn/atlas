import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent.api.routes import health
from agent.config import Settings
from agent.services.ollama import OllamaClient


def test_health_reports_configured_models(tmp_path) -> None:
    settings = _settings(tmp_path, chat_model="qwen3:4b", embedding_model="embeddinggemma:latest")
    request = SimpleNamespace(
        app=SimpleNamespace(
            version="0.2.0",
            state=SimpleNamespace(settings=settings),
        )
    )

    response = asyncio.run(health(request))

    assert response == {
        "status": "ok",
        "version": "0.2.0",
        "models": {
            "chat": "qwen3:4b",
            "embedding": "embeddinggemma:latest",
        },
    }


def test_ollama_chat_uses_configured_model_and_generation_options() -> None:
    client = RecordingOllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen3:4b",
        embedding_model="embeddinggemma:latest",
    )

    response = asyncio.run(client.chat([{"role": "user", "content": "hello"}]))

    assert response == "ok"
    assert client.calls[0]["path"] == "/api/chat"
    payload = client.calls[0]["payload"]
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "15m"
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "num_predict": 300,
        "num_ctx": 4096,
    }
    assert "format" not in payload


def test_ollama_chat_uses_json_mode_when_requested() -> None:
    client = RecordingOllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen3:4b",
        embedding_model="embeddinggemma:latest",
    )

    asyncio.run(client.chat([{"role": "user", "content": "hello"}], json_mode=True))

    payload = client.calls[0]["payload"]
    assert payload["format"] == "json"


def test_ollama_embedding_uses_latest_embedding_model() -> None:
    client = RecordingOllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen3:4b",
        embedding_model="embeddinggemma:latest",
    )

    embedding = asyncio.run(client.embed("evidence"))

    assert embedding == [1.0, 0.0]
    assert client.calls[0]["path"] == "/api/embeddings"
    assert client.calls[0]["payload"]["model"] == "embeddinggemma:latest"


def test_production_code_does_not_select_qwen3_8b() -> None:
    root = Path(__file__).resolve().parents[1]
    production_files = [
        *root.glob("*.py"),
        *Path(root / "agent").rglob("*.py"),
        root / "README.md",
        root / ".env.example",
    ]

    offenders = [
        str(path.relative_to(root))
        for path in production_files
        if "qwen3:8b" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


class RecordingOllamaClient(OllamaClient):
    def __init__(self, base_url: str, chat_model: str, embedding_model: str) -> None:
        super().__init__(base_url, chat_model, embedding_model)
        self.calls: list[dict[str, object]] = []

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append({"path": path, "payload": payload})
        if path == "/api/chat":
            return {"message": {"content": "ok"}}
        return {"embedding": [1.0, 0.0]}


def _settings(tmp_path: Path, chat_model: str, embedding_model: str) -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434",
        chat_model=chat_model,
        embedding_model=embedding_model,
        database_path=tmp_path / "atlas.db",
        obsidian_vault_path=tmp_path,
        default_top_k=5,
        min_top_score=0.35,
        min_average_score=0.25,
        min_meaningful_content_chars=40,
    )
