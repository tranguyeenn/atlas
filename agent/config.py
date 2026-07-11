from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


OBSIDIAN_VAULT_PATH_HELP = (
    "OBSIDIAN_VAULT_PATH is not configured. Open your vault in Obsidian, reveal "
    "or open the vault folder in your file manager, then copy the folder path. "
    "On macOS, right-click the vault folder in Finder, hold Option, and choose "
    "'Copy ... as Pathname'. Add that path to .env as OBSIDIAN_VAULT_PATH, or run: "
    "python3 -m agent.cli configure-obsidian"
)


class ObsidianVaultPathError(ValueError):
    pass


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    database_path: Path
    obsidian_vault_path: Path | None
    default_top_k: int


def get_settings() -> Settings:
    _load_dotenv(Path(".env"))

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    database_path = Path(os.getenv("DATABASE_PATH", "data/atlas.db"))

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        database_path=database_path,
        obsidian_vault_path=Path(vault_path).expanduser() if vault_path else None,
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
    )


def validate_obsidian_vault_path(vault_path: Path | None) -> Path:
    if vault_path is None:
        raise ObsidianVaultPathError(OBSIDIAN_VAULT_PATH_HELP)

    expanded_path = vault_path.expanduser()
    if not expanded_path.exists() or not expanded_path.is_dir():
        raise ObsidianVaultPathError(
            f"Invalid OBSIDIAN_VAULT_PATH: {expanded_path}. "
            "The path must point to an existing Obsidian vault folder."
        )

    return expanded_path
