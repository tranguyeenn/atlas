from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ObsidianVaultPathError,
    get_settings,
    validate_obsidian_vault_path,
)
from agent.models.schemas import IndexRequest


def test_get_settings_uses_default_models_without_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)

    settings = get_settings()

    assert settings.chat_model == DEFAULT_CHAT_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.min_top_score == 0.35
    assert settings.min_average_score == 0.25
    assert settings.min_meaningful_content_chars == 40


def test_get_settings_loads_model_from_env_file_and_overrides_stale_env(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "stale-model")
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "OLLAMA_CHAT_MODEL=qwen3:4b\n"
        "OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest\n",
        encoding="utf-8",
    )

    settings = get_settings()

    assert settings.chat_model == "qwen3:4b"
    assert settings.embedding_model == "embeddinggemma:latest"


def test_get_settings_resolves_active_vault_path_from_env_file(tmp_path, monkeypatch) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f'OBSIDIAN_VAULT_PATH="{vault_path}"\n',
        encoding="utf-8",
    )

    settings = get_settings()

    assert settings.obsidian_vault_path == vault_path
    assert validate_obsidian_vault_path(settings.obsidian_vault_path) == vault_path


def test_index_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        IndexRequest.model_validate({"path": "cosmos", "force": True})


def test_index_request_accepts_force_reindex_payload() -> None:
    payload = IndexRequest.model_validate({"vault_path": None, "force": True})

    assert payload.vault_path is None
    assert payload.force is True


def test_validate_obsidian_vault_path_requires_configuration() -> None:
    with pytest.raises(ObsidianVaultPathError) as exc_info:
        validate_obsidian_vault_path(None)

    message = str(exc_info.value)
    assert "OBSIDIAN_VAULT_PATH is not configured" in message
    assert "python3 -m agent.cli configure-obsidian" in message


def test_validate_obsidian_vault_path_shows_invalid_path(tmp_path) -> None:
    missing_path = tmp_path / "missing-vault"

    with pytest.raises(ObsidianVaultPathError) as exc_info:
        validate_obsidian_vault_path(missing_path)

    assert str(missing_path) in str(exc_info.value)


def test_validate_obsidian_vault_path_accepts_existing_directory(tmp_path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    assert validate_obsidian_vault_path(vault_path) == vault_path


def test_validate_obsidian_vault_path_rejects_file(tmp_path) -> None:
    vault_path = tmp_path / "note.md"
    vault_path.write_text("# Note\n", encoding="utf-8")

    with pytest.raises(ObsidianVaultPathError) as exc_info:
        validate_obsidian_vault_path(Path(vault_path))

    assert str(vault_path) in str(exc_info.value)
