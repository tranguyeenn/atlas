from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

from agent.services.ollama import OllamaClient


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    filename: str
    path: str
    heading: str
    content: str
    score: float


class SemanticSearch:
    def __init__(self, connection: sqlite3.Connection, ollama: OllamaClient) -> None:
        self.connection = connection
        self.ollama = ollama

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = await self.ollama.embed(query)
        rows = self.connection.execute(
            """
            SELECT chunks.id AS chunk_id, files.filename, files.path, chunks.heading,
                   chunks.content, chunks.embedding
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            """
        ).fetchall()

        scored: list[RetrievedChunk] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, [float(value) for value in embedding])
            scored.append(
                RetrievedChunk(
                    chunk_id=int(row["chunk_id"]),
                    filename=row["filename"],
                    path=row["path"],
                    heading=row["heading"],
                    content=row["content"],
                    score=score,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
