import sqlite3

import pytest

from agent.database import connect, initialize_database


def test_initialize_database_creates_entity_graph_tables(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"

    initialize_database(database_path)

    with connect(database_path) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "entities" in tables
    assert "entity_relationships" in tables


def test_entities_enforce_supported_entity_types(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    initialize_database(database_path)

    with connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO entities(type, name, created_at, updated_at)
            VALUES ('Project', 'Atlas', '2026-07-14T00:00:00+00:00', '2026-07-14T00:00:00+00:00')
            """
        )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO entities(type, name, created_at, updated_at)
                VALUES ('Chunk', 'Implementation Detail', '2026-07-14T00:00:00+00:00', '2026-07-14T00:00:00+00:00')
                """
            )


def test_entity_relationships_can_reference_source_evidence(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    initialize_database(database_path)

    with connect(database_path) as connection:
        file_id = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES ('/vault/atlas.md', 'atlas.md', 'hash', '2026-07-14T00:00:00+00:00')
            """
        ).lastrowid
        chunk_id = connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, 'Decisions', 'Atlas should be entity-first.', '[1.0]', 0, '2026-07-14T00:00:00+00:00')
            """,
            (file_id,),
        ).lastrowid
        project_id = connection.execute(
            """
            INSERT INTO entities(type, name, created_at, updated_at)
            VALUES ('Project', 'Atlas', '2026-07-14T00:00:00+00:00', '2026-07-14T00:00:00+00:00')
            """
        ).lastrowid
        decision_id = connection.execute(
            """
            INSERT INTO entities(type, name, created_at, updated_at)
            VALUES ('Decision', 'Entity-first model', '2026-07-14T00:00:00+00:00', '2026-07-14T00:00:00+00:00')
            """
        ).lastrowid

        connection.execute(
            """
            INSERT INTO entity_relationships(
                source_entity_id,
                target_entity_id,
                type,
                evidence_chunk_id,
                created_at
            )
            VALUES (?, ?, 'Decision -> Project', ?, '2026-07-14T00:00:00+00:00')
            """,
            (decision_id, project_id, chunk_id),
        )
        relationship = connection.execute(
            """
            SELECT relationships.type, chunks.content
            FROM entity_relationships AS relationships
            JOIN chunks ON chunks.id = relationships.evidence_chunk_id
            """
        ).fetchone()

    assert relationship["type"] == "Decision -> Project"
    assert relationship["content"] == "Atlas should be entity-first."
