import json
import asyncio

from agent.database import connect, initialize_database
from agent.retrieval.search import SemanticSearch, cosine_similarity


class FakeOllama:
    async def embed(self, text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_cosine_similarity_scores_expected_values() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_semantic_search_orders_by_similarity(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    initialize_database(database_path)

    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("note-a.md", "note-a.md", "hash-a", "now"),
        )
        file_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, "Alpha", "alpha content", json.dumps([1.0, 0.0]), 0, "now"),
        )
        connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, "Beta", "beta content", json.dumps([0.0, 1.0]), 1, "now"),
        )
        connection.commit()

        results = asyncio.run(SemanticSearch(connection, FakeOllama()).search("alpha query", top_k=2))

    assert [result.heading for result in results] == ["Alpha", "Beta"]
