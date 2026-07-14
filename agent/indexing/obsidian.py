from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.indexing.chunker import chunk_markdown_by_headings
from agent.indexing.entities import IndexedChunk, delete_entities_for_file, extract_entities_for_file
from agent.services.ollama import OllamaClient


@dataclass(frozen=True)
class IndexStats:
    indexed_files: int = 0
    skipped_files: int = 0
    indexed_chunks: int = 0


class ObsidianIndexer:
    def __init__(self, connection: sqlite3.Connection, ollama: OllamaClient) -> None:
        self.connection = connection
        self.ollama = ollama

    async def index_vault(self, vault_path: Path, force: bool = False) -> IndexStats:
        if not vault_path.exists() or not vault_path.is_dir():
            raise ValueError(f"Vault path does not exist or is not a directory: {vault_path}")

        indexed_files = 0
        skipped_files = 0
        indexed_chunks = 0

        for markdown_file in sorted(vault_path.rglob("*.md")):
            if _is_hidden_path(markdown_file, vault_path):
                continue
            file_chunks = await self.index_file(markdown_file, force=force)
            if file_chunks is None:
                skipped_files += 1
                continue
            indexed_files += 1
            indexed_chunks += file_chunks

        return IndexStats(
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            indexed_chunks=indexed_chunks,
        )

    async def index_file(self, path: Path, force: bool = False) -> int | None:
        content = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self.connection.execute(
            "SELECT id, content_hash FROM files WHERE path = ?",
            (str(path),),
        ).fetchone()

        if existing and existing["content_hash"] == content_hash and not force:
            return None

        chunks = chunk_markdown_by_headings(content)
        now = datetime.now(timezone.utc).isoformat()

        with self.connection:
            if existing:
                file_id = int(existing["id"])
                delete_entities_for_file(self.connection, file_id)
                self.connection.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                self.connection.execute(
                    """
                    UPDATE files
                    SET filename = ?, content_hash = ?, indexed_at = ?
                    WHERE id = ?
                    """,
                    (path.name, content_hash, now, file_id),
                )
            else:
                cursor = self.connection.execute(
                    """
                    INSERT INTO files(path, filename, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(path), path.name, content_hash, now),
                )
                file_id = int(cursor.lastrowid)

            indexed_chunks: list[IndexedChunk] = []
            for index, chunk in enumerate(chunks):
                embedding = await self.ollama.embed(chunk.content)
                cursor = self.connection.execute(
                    """
                    INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (file_id, chunk.heading, chunk.content, json.dumps(embedding), index, now),
                )
                indexed_chunks.append(
                    IndexedChunk(
                        id=int(cursor.lastrowid),
                        heading=chunk.heading,
                        content=chunk.content,
                        chunk_index=index,
                    )
                )

            extract_entities_for_file(
                self.connection,
                file_id=file_id,
                path=path,
                chunks=indexed_chunks,
            )

        return len(chunks)


def _is_hidden_path(path: Path, vault_path: Path) -> bool:
    relative_parts = path.relative_to(vault_path).parts
    return any(part.startswith(".") for part in relative_parts)
