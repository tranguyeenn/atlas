import asyncio

from agent.database import connect, initialize_database
from agent.indexing.obsidian import ObsidianIndexer
from agent.retrieval.search import SemanticSearch
from agent.services.project_state import ProjectStateIntent, ProjectStateService


class FakeOllama:
    async def embed(self, text: str) -> list[float]:
        if "First" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_obsidian_indexing_still_indexes_markdown_chunks(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "note.md").write_text("# First\nalpha\n\n## Second\nbeta\n", encoding="utf-8")
    initialize_database(database_path)

    with connect(database_path) as connection:
        stats = asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        rows = connection.execute(
            """
            SELECT files.filename, chunks.heading, chunks.content
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            ORDER BY chunks.chunk_index
            """
        ).fetchall()

    assert stats.indexed_files == 1
    assert stats.skipped_files == 0
    assert stats.indexed_chunks == 2
    assert [row["filename"] for row in rows] == ["note.md", "note.md"]
    assert [row["heading"] for row in rows] == ["First", "Second"]


def test_obsidian_indexing_preserves_checkbox_markers_in_retrieved_content(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "tasks.md").write_text(
        "# Tasks\n- [x] completed\n- [ ] incomplete\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        results = asyncio.run(SemanticSearch(connection, FakeOllama()).search("completed", top_k=1))

    assert "- [x] completed" in results[0].content
    assert "- [ ] incomplete" in results[0].content


def test_obsidian_indexing_extracts_project_and_task_entities(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "The Long Orbit.md").write_text(
        "\n".join(
            [
                "# The Long Orbit",
                "## Status",
                "In progress.",
                "## Phase",
                "Learning foundations.",
                "## Project Goals",
                "- Model long-term planetary habitability.",
                "## Checklist",
                "- [x] Define habitability",
                "- [ ] Learn stellar evolution basics",
            ]
        ),
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        projects = connection.execute(
            "SELECT type, name FROM entities WHERE type = 'Project'"
        ).fetchall()
        tasks = connection.execute(
            """
            SELECT tasks.description, state.value AS state
            FROM entities AS tasks
            JOIN entity_attributes AS state
              ON state.entity_id = tasks.id AND state.key = 'state'
            WHERE tasks.type = 'Task'
            ORDER BY tasks.description
            """
        ).fetchall()
        relationships = connection.execute(
            "SELECT type FROM entity_relationships WHERE type = 'Project -> Task'"
        ).fetchall()

    assert [(row["type"], row["name"]) for row in projects] == [("Project", "The Long Orbit")]
    assert [(row["description"], row["state"]) for row in tasks] == [
        ("Define habitability", "completed"),
        ("Learn stellar evolution basics", "incomplete"),
    ]
    assert len(relationships) == 2


def test_project_folder_name_becomes_project_owner_and_phase_heading_is_not_project(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# Phase 1: Define Habitability\n\n## Current Phase Checklist\n- Define habitability\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        projects = connection.execute(
            "SELECT name FROM entities WHERE type = 'Project' ORDER BY name"
        ).fetchall()
        linked_tasks = connection.execute(
            """
            SELECT projects.name AS project_name, tasks.description
            FROM entity_relationships AS relationships
            JOIN entities AS projects ON projects.id = relationships.source_entity_id
            JOIN entities AS tasks ON tasks.id = relationships.target_entity_id
            WHERE relationships.type = 'Project -> Task'
            """
        ).fetchall()

    assert [row["name"] for row in projects] == ["The Long Orbit"]
    assert [(row["project_name"], row["description"]) for row in linked_tasks] == [
        ("The Long Orbit", "Define habitability")
    ]


def test_current_phase_checklist_plain_bullets_become_ordered_unknown_tasks(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "\n".join(
            [
                "# The Long Orbit",
                "## Current Phase Checklist",
                "- Define habitability",
                "- Understand habitable zones",
                "- Learn stellar evolution basics",
            ]
        ),
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        tasks = _task_rows(connection)

    assert [(row["description"], row["state"], row["ordinal"]) for row in tasks] == [
        ("Define habitability", "unknown", "0"),
        ("Understand habitable zones", "unknown", "1"),
        ("Learn stellar evolution basics", "unknown", "2"),
    ]


def test_plain_bullets_outside_task_oriented_sections_do_not_become_tasks(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Major Topics\n- Stellar evolution\n- Habitability\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        task_count = connection.execute(
            "SELECT COUNT(*) AS count FROM entities WHERE type = 'Task'"
        ).fetchone()["count"]

    assert task_count == 0


def test_checkbox_states_map_to_completed_and_incomplete(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Checklist\n- [x] Define habitability\n- [ ] Learn stellar evolution basics\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        tasks = _task_rows(connection)

    assert [(row["description"], row["state"]) for row in tasks] == [
        ("Define habitability", "completed"),
        ("Learn stellar evolution basics", "incomplete"),
    ]


def test_unknown_plain_list_tasks_are_eligible_for_next_task_resolution(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Current Phase Checklist\n- Define habitability\n- Understand habitable zones\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What should I work on next in The Long Orbit?",
            ProjectStateIntent.NEXT_TASK,
        )

    assert response is not None
    assert response.category == "project_state"
    assert response.recommended_action == "Define habitability"
    assert response.sources[0].file == "Project Overview.md"
    assert response.sources[0].heading == "Current Phase Checklist"


def test_explicit_current_phase_is_extracted_and_status_is_preserved(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Status\nActive\n\n## Current Phase\nFoundation Building Phase\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        attrs = _project_attrs(connection, "The Long Orbit")

    assert attrs["status"] == "Active"
    assert attrs["phase"] == "Foundation Building Phase"


def test_explicit_project_phase_takes_precedence_over_phase_like_status(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "\n".join(
            [
                "# The Long Orbit",
                "## Status",
                "Planning Phase",
                "## Project Phase",
                "Foundation Building Phase",
            ]
        ),
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What phase is The Long Orbit in?",
            ProjectStateIntent.PHASE,
        )

    assert response is not None
    assert response.answer == "The Long Orbit is in phase: Foundation Building Phase"


def test_phase_like_status_is_used_as_effective_phase_fallback(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Status\nFoundation Building Phase\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What phase is The Long Orbit in?",
            ProjectStateIntent.PHASE,
        )
        attrs = _project_attrs(connection, "The Long Orbit")

    assert response is not None
    assert response.answer == "The Long Orbit is in phase: Foundation Building Phase"
    assert attrs["status"] == "Foundation Building Phase"
    assert "phase" not in attrs


def test_generic_status_active_is_not_used_as_effective_phase(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Status\nActive\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What phase is The Long Orbit in?",
            ProjectStateIntent.PHASE,
        )

    assert response is not None
    assert response.answer == "The Long Orbit has no indexed phase"
    assert response.missing_information == ["The Long Orbit has no indexed phase."]


def test_next_task_answer_uses_effective_phase_and_hides_unknown_state(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Status\nFoundation Building Phase\n\n## Current Phase Checklist\n- Define habitability\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What should I work on next in The Long Orbit?",
            ProjectStateIntent.NEXT_TASK,
        )

    assert response is not None
    assert response.answer == (
        "Work on this next for The Long Orbit: Define habitability. "
        "Current phase: Foundation Building Phase."
    )
    assert "Task state: unknown" not in response.answer


def test_unfinished_work_response_uses_task_display_names(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Current Phase Checklist\n- Define habitability\n- Understand habitable zones\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path))
        response = ProjectStateService(connection).answer(
            "What remains unfinished in The Long Orbit?",
            ProjectStateIntent.UNFINISHED,
        )

    assert response is not None
    assert response.answer == "Unfinished work: Define habitability (unknown); Understand habitable zones (unknown)."
    assert "The Long Orbit: Define habitability" not in response.answer


def test_forced_reindex_removes_stale_incorrectly_linked_entities(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    inbox_dir = vault_path / "00 Inbox"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    inbox_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    inbox_file = inbox_dir / "Overview.md"
    project_file = project_dir / "Project Overview.md"
    inbox_file.write_text("# Phase 1: Define Habitability\n\n## Checklist\n- Create fake task\n", encoding="utf-8")
    project_file.write_text("# The Long Orbit\n\n## Current Phase Checklist\n- Define habitability\n", encoding="utf-8")
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        inbox_projects = connection.execute(
            "SELECT name FROM entities WHERE type = 'Project' AND name = 'Phase 1: Define Habitability'"
        ).fetchall()
        first_task_count = connection.execute(
            "SELECT COUNT(*) AS count FROM entities WHERE type = 'Task'"
        ).fetchone()["count"]
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        second_task_count = connection.execute(
            "SELECT COUNT(*) AS count FROM entities WHERE type = 'Task'"
        ).fetchone()["count"]
        relationships = connection.execute(
            "SELECT COUNT(*) AS count FROM entity_relationships WHERE type = 'Project -> Task'"
        ).fetchone()["count"]

    assert inbox_projects == []
    assert first_task_count == 1
    assert second_task_count == 1
    assert relationships == 1


def test_inbox_phase_heading_does_not_create_fake_project_entity(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    inbox_dir = vault_path / "00 Inbox"
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "Overview.md").write_text(
        "# Phase 1: Define Habitability\n\n## Checklist\n- Create Habitability.md\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        entities = connection.execute(
            "SELECT type, name FROM entities ORDER BY type, name"
        ).fetchall()

    assert [(row["type"], row["name"]) for row in entities] == []


def test_repeated_indexing_remains_idempotent_for_entities(tmp_path) -> None:
    database_path = tmp_path / "atlas.db"
    vault_path = tmp_path / "vault"
    project_dir = vault_path / "09 Projects" / "The Long Orbit"
    project_dir.mkdir(parents=True)
    (project_dir / "Project Overview.md").write_text(
        "# The Long Orbit\n\n## Current Phase Checklist\n- Define habitability\n- Understand habitable zones\n",
        encoding="utf-8",
    )
    initialize_database(database_path)

    with connect(database_path) as connection:
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        first = _entity_counts(connection)
        asyncio.run(ObsidianIndexer(connection, FakeOllama()).index_vault(vault_path, force=True))
        second = _entity_counts(connection)

    assert first == second == {"Project": 1, "Task": 2}


def _task_rows(connection):
    return connection.execute(
        """
        SELECT tasks.description, state.value AS state, ordinal.value AS ordinal
        FROM entities AS tasks
        JOIN entity_attributes AS state
          ON state.entity_id = tasks.id AND state.key = 'state'
        JOIN entity_attributes AS ordinal
          ON ordinal.entity_id = tasks.id AND ordinal.key = 'ordinal'
        WHERE tasks.type = 'Task'
        ORDER BY CAST(ordinal.value AS INTEGER)
        """
    ).fetchall()


def _entity_counts(connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT type, COUNT(*) AS count FROM entities GROUP BY type"
    ).fetchall()
    return {row["type"]: row["count"] for row in rows}


def _project_attrs(connection, project_name: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT attrs.key, attrs.value
        FROM entities AS projects
        JOIN entity_attributes AS attrs ON attrs.entity_id = projects.id
        WHERE projects.type = 'Project' AND projects.name = ?
        """,
        (project_name,),
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}
