from __future__ import annotations

from pathlib import Path

from agent.models.schemas import ResearchSource
from agent.retrieval.search import RetrievedChunk


def format_sources(
    chunks: list[RetrievedChunk],
    vault_path: Path | None = None,
) -> list[ResearchSource]:
    return [
        ResearchSource(
            id=f"source_{index}",
            file=chunk.filename,
            path=_relative_path(chunk.path, vault_path),
            heading=chunk.heading or None,
            score=chunk.score,
            excerpt=_excerpt(chunk.content),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _relative_path(path: str, vault_path: Path | None) -> str:
    candidate = Path(path)
    if vault_path is None or not candidate.is_absolute():
        return path

    try:
        return str(candidate.relative_to(vault_path))
    except ValueError:
        return path


def _excerpt(content: str, max_length: int = 280) -> str:
    text = " ".join(content.split())
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."
