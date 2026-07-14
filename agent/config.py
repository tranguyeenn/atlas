from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


OBSIDIAN_VAULT_PATH_HELP = (
    "OBSIDIAN_VAULT_PATH is not configured. Open your vault in Obsidian, reveal "
    "or open the vault folder in your file manager, then copy the folder path. "
    "On macOS, right-click the vault folder in Finder, hold Option, and choose "
    "'Copy ... as Pathname'. Add that path to .env as OBSIDIAN_VAULT_PATH, or run: "
    "python3 -m agent.cli configure-obsidian"
)


class ObsidianVaultPathError(ValueError):
    pass


DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"


def load_environment(env_file: Path = Path(".env")) -> None:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    database_path: Path
    obsidian_vault_path: Path | None
    default_top_k: int
    min_top_score: float
    min_average_score: float
    min_meaningful_content_chars: int


def get_settings() -> Settings:
    load_environment()

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    database_path = Path(os.getenv("DATABASE_PATH", "data/atlas.db"))

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        database_path=database_path,
        obsidian_vault_path=Path(vault_path).expanduser() if vault_path else None,
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        min_top_score=float(os.getenv("MIN_TOP_SCORE", "0.35")),
        min_average_score=float(os.getenv("MIN_AVERAGE_SCORE", "0.25")),
        min_meaningful_content_chars=int(os.getenv("MIN_MEANINGFUL_CONTENT_CHARS", "40")),
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
