from pathlib import Path

import pytest

from agent.config import ObsidianVaultPathError, validate_obsidian_vault_path


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
