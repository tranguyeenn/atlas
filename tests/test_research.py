import asyncio
import json
from pathlib import Path

from agent.api.routes import research, search_notes
from agent.config import Settings
from agent.database import connect, initialize_database
from agent.models.schemas import (
    ResearchBrief,
    ResearchErrorResponse,
    ProjectStateResponse,
    ResearchRequest,
    ResearchRedirectResponse,
    ResearchSource,
    ResearchUnsupportedResponse,
    SearchRequest,
)
from agent.services.research_brief import (
    ResearchBriefService,
    parse_llm_research_brief,
    validate_source_references,
)


class FakeOllama:
    def __init__(self, chat_responses: list[str] | None = None) -> None:
        self.chat_responses = chat_responses or []
        self.chat_calls = 0
        self.embed_calls = 0
        self.chat_json_modes: list[bool] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        if "French Revolution" in text:
            return [0.0, 0.0]
        if "remains" in text or "completed" in text:
            return [0.7, 0.3]
        if "goal" in text:
            return [1.0, 0.0]
        if "nonexistent" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    async def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        self.chat_calls += 1
        self.chat_json_modes.append(json_mode)
        if self.chat_responses:
            return self.chat_responses.pop(0)
        return json.dumps(
            {
                "key_points": [
                    {
                        "text": "The note links orbital modeling to habitability evidence.",
                        "source_ids": ["source_1"],
                    }
                ],
                "connections": [
                    {
                        "concept": "Orbital mechanics",
                        "explanation": "The source frames orbital mechanics as the model basis.",
                        "source_ids": ["source_1"],
                    }
                ],
                "open_questions": ["Which variables matter most?"],
                "missing_information": ["The notes do not name an integration method."],
            }
        )


def test_research_assistance_returns_structured_brief(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_chunk(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(
                question="How does The Long Orbit connect mathematics and astronomy?",
                top_k=1,
            ),
            settings,
            FakeOllama(),
        )
    )

    assert isinstance(response, ResearchBrief)
    assert response.status == "ok"
    assert response.question == "How does The Long Orbit connect mathematics and astronomy?"
    assert response.sources[0].id == "source_1"
    assert response.sources[0].file == "The Long Orbit.md"
    assert response.key_points[0].source_ids == ["source_1"]
    assert response.connections[0].source_ids == ["source_1"]


def test_obvious_generative_writing_request_redirects_without_ollama_calls(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    fake = FakeOllama()

    response = asyncio.run(
        research(
            ResearchRequest(question="Write my complete research paper about habitability."),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchRedirectResponse)
    assert response.status == "redirect"
    assert response.category == "generative_writing"
    assert fake.chat_calls == 0
    assert fake.embed_calls == 0


def test_unsupported_request_with_no_relevant_indexed_notes(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_chunk(settings.database_path)
    fake = FakeOllama()

    response = asyncio.run(
        research(
            ResearchRequest(question="What does my notes say about nonexistent materials?"),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchUnsupportedResponse)
    assert response.status == "unsupported"
    assert fake.chat_calls == 0


def test_french_revolution_returns_unsupported_without_generation(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_chunks(settings.database_path)
    fake = FakeOllama()

    response = asyncio.run(
        research(
            ResearchRequest(question="Explain the French Revolution.", top_k=3),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchUnsupportedResponse)
    assert response.status == "unsupported"
    assert response.question == "Explain the French Revolution."
    assert response.message == "Atlas could not find relevant evidence in the indexed notes."
    assert fake.chat_calls == 0


def test_research_goal_request_remains_supported(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_chunks(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What is the goal of The Long Orbit?", top_k=2),
            settings,
            FakeOllama(),
        )
    )

    assert isinstance(response, ResearchBrief)
    assert response.sources[0].heading == "Project Goals"


def test_next_task_request_uses_project_state_not_goal_retrieval(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_project_state(settings.database_path)
    fake = FakeOllama()

    response = asyncio.run(
        research(
            ResearchRequest(question="What should I work on next in The Long Orbit?", top_k=3),
            settings,
            fake,
        )
    )

    assert isinstance(response, ProjectStateResponse)
    assert response.category == "project_state"
    assert response.recommended_action == "Learn stellar evolution basics"
    assert "long-term planetary habitability" not in response.answer
    assert fake.chat_calls == 0
    assert fake.embed_calls == 0


def test_active_projects_are_aggregated_from_project_entities(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_project_state(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What projects are active?", top_k=3),
            settings,
            FakeOllama(),
        )
    )

    assert isinstance(response, ProjectStateResponse)
    assert response.answer.startswith("Active projects:")
    assert response.entities[0].name == "The Long Orbit"
    assert response.entities[0].type == "Project"


def test_completion_request_prioritizes_checklist_sources(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_chunks(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed in The Long Orbit?", top_k=2),
            settings,
            FakeOllama(),
        )
    )

    assert isinstance(response, ResearchBrief)
    assert response.sources[0].heading in {"Status", "Checklist"}


def test_roadmap_completion_excludes_learn_and_includes_deliverables_and_criteria(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_roadmap_chunks(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed?", top_k=4),
            settings,
            FakeOllama(chat_responses=[json.dumps(_brief_payload())]),
        )
    )

    assert isinstance(response, ResearchBrief)
    headings = [source.heading for source in response.sources]
    assert "Deliverables" in headings
    assert "Completion Criteria" in headings
    assert "Learn" not in headings[:3]
    combined_points = "\n".join(item.text for item in response.key_points)
    assert "Future deliverables: Graph of habitable-zone movement" in combined_points
    assert "Upcoming milestones: Can explain habitable zone migration" in combined_points
    assert "Stellar luminosity" not in combined_points


def test_status_evidence_precedes_deliverables_for_completion_query(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_roadmap_chunks(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed?", top_k=4),
            settings,
            FakeOllama(chat_responses=[json.dumps(_brief_payload())]),
        )
    )

    assert isinstance(response, ResearchBrief)
    assert response.sources[0].heading == "Status"
    assert response.key_points[0].text.startswith("Current status:")
    assert "No simulation development has started" in response.key_points[0].text


def test_completed_roadmap_deliverables_are_excluded_while_neutral_remain(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_mixed_deliverables_chunks(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed?", top_k=3),
            settings,
            FakeOllama(chat_responses=[json.dumps(_brief_payload())]),
        )
    )

    assert isinstance(response, ResearchBrief)
    combined_points = "\n".join(item.text for item in response.key_points)
    assert "Future deliverables: Graph of habitable zone movement; Initial model documentation" in combined_points
    assert "Project overview completed" not in combined_points
    assert "Initial journal entry completed" not in combined_points


def test_duplicate_deterministic_and_llm_status_key_points_are_removed(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_roadmap_chunks(settings.database_path)
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [
                        {
                            "text": "The project has no simulation development started.",
                            "source_ids": ["source_1"],
                        }
                    ],
                    "connections": [],
                    "open_questions": [],
                    "missing_information": [],
                }
            )
        ]
    )

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed?", top_k=4),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    status_points = [
        item.text
        for item in response.key_points
        if "simulation development" in item.text.lower()
    ]
    assert status_points == ["Current status: No simulation development has started."]
    assert all(source_id in {source.id for source in response.sources} for item in response.key_points for source_id in item.source_ids)


def test_distinct_llm_finding_from_same_source_is_preserved(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_roadmap_chunks(settings.database_path)
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [
                        {
                            "text": "The status source also implies implementation planning is still needed.",
                            "source_ids": ["source_1"],
                        }
                    ],
                    "connections": [],
                    "open_questions": [],
                    "missing_information": [],
                }
            )
        ]
    )

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed?", top_k=4),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchBrief)
    combined_points = "\n".join(item.text for item in response.key_points)
    assert "Current status: No simulation development has started." in combined_points
    assert "implementation planning is still needed" in combined_points


def test_completion_answer_excludes_completed_tasks_and_includes_incomplete(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_task_state_chunks(settings.database_path)
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [
                        {
                            "text": "Rename `cosmos-lab` to `The Long Orbit` remains incomplete.",
                            "source_ids": ["source_1"],
                        }
                    ],
                    "connections": [],
                    "open_questions": [],
                    "missing_information": [],
                }
            )
        ]
    )

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed in The Long Orbit?", top_k=2),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    combined_points = "\n".join(item.text for item in response.key_points)
    assert "Remaining: Create a graph of habitable-zone movement" in combined_points
    assert "Completion status unclear: Initial model documentation" in combined_points
    assert "Rename `cosmos-lab` to `The Long Orbit` remains incomplete" not in combined_points
    assert "Create `Project Overview.md`" not in combined_points


def test_explicit_not_started_status_is_incomplete_evidence(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_status_evidence_chunk(settings.database_path)

    response = asyncio.run(
        research(
            ResearchRequest(question="What remains to be completed in The Long Orbit?", top_k=1),
            settings,
            FakeOllama(chat_responses=[json.dumps(_brief_payload())]),
        )
    )

    assert isinstance(response, ResearchBrief)
    assert response.key_points[0].text.startswith("Current status:")
    assert "No simulation development has started" in response.key_points[0].text


def test_empty_absence_brief_becomes_unsupported(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_long_orbit_chunks(settings.database_path)
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [],
                    "connections": [],
                    "open_questions": [],
                    "missing_information": [
                        "Information about the French Revolution is absent from the retrieved context."
                    ],
                }
            )
        ]
    )

    response = asyncio.run(
        research(
            ResearchRequest(question="What is the goal of The Long Orbit?", top_k=2),
            settings,
            fake,
        )
    )

    assert isinstance(response, ResearchUnsupportedResponse)
    assert response.question == "What is the goal of The Long Orbit?"


def test_invalid_source_ids_are_removed() -> None:
    sources = [
        ResearchSource(
            id="source_1",
            file="note.md",
            path="note.md",
            heading="Method",
            score=0.9,
            excerpt="Evidence",
        )
    ]
    parsed, error = parse_llm_research_brief(
        json.dumps(
            {
                "key_points": [
                    {"text": "Valid cited point", "source_ids": ["source_1", "source_99"]},
                    {"text": "Invalid cited point", "source_ids": ["source_99"]},
                ],
                "connections": [
                    {
                        "concept": "Modeling",
                        "explanation": "Valid connection",
                        "source_ids": ["source_1"],
                    },
                    {
                        "concept": "Unsupported",
                        "explanation": "Invalid connection",
                        "source_ids": ["source_99"],
                    },
                ],
                "open_questions": [],
                "missing_information": [],
            }
        )
    )

    assert error is None
    assert parsed is not None
    sanitized = validate_source_references(parsed, sources)
    assert [item.source_ids for item in sanitized.key_points] == [["source_1"]]
    assert [item.source_ids for item in sanitized.connections] == [["source_1"]]


def test_malformed_llm_json_triggers_repair() -> None:
    source = _source()
    fake = FakeOllama(
        chat_responses=[
            "not json",
            json.dumps(
                {
                    "key_points": [{"text": "Repaired point", "source_ids": ["source_1"]}],
                    "connections": [
                        {
                            "concept": "Repair",
                            "explanation": "The repaired JSON cites the source.",
                            "source_ids": ["source_1"],
                        }
                    ],
                    "open_questions": [],
                    "missing_information": [],
                }
            ),
        ]
    )

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [source],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 2
    assert fake.chat_json_modes == [True, True]
    assert response.key_points[0].source_ids == ["source_1"]


def test_malformed_json_after_repair_returns_controlled_failure() -> None:
    fake = FakeOllama(chat_responses=["not json", "still not json"])

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchErrorResponse)
    assert response.status == "error"
    assert fake.chat_calls == 2


def test_valid_first_response_avoids_repair() -> None:
    fake = FakeOllama()

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    assert fake.chat_json_modes == [True]


def test_fenced_json_is_cleaned_without_repair() -> None:
    payload = _brief_payload()
    fake = FakeOllama(chat_responses=[f"```json\n{json.dumps(payload)}\n```"])

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    assert response.key_points[0].source_ids == ["source_1"]


def test_harmless_prose_around_json_is_cleaned_without_repair() -> None:
    payload = _brief_payload()
    fake = FakeOllama(chat_responses=[f"Here is the JSON:\n{json.dumps(payload)}\nDone."])

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1


def test_missing_optional_arrays_default_to_empty_without_repair() -> None:
    fake = FakeOllama(chat_responses=[json.dumps({"key_points": []})])

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    assert response.key_points == []
    assert response.connections == []
    assert response.open_questions == []
    assert response.missing_information == []


def test_invalid_source_ids_are_filtered_without_repair() -> None:
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [
                        {"text": "Mixed citations", "source_ids": ["source_1", "source_99"]}
                    ],
                    "connections": [],
                    "open_questions": [],
                    "missing_information": [],
                }
            )
        ]
    )

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    assert response.key_points[0].source_ids == ["source_1"]


def test_uncited_items_are_dropped_without_repair() -> None:
    fake = FakeOllama(
        chat_responses=[
            json.dumps(
                {
                    "key_points": [{"text": "Uncited", "source_ids": []}],
                    "connections": [
                        {"concept": "No citation", "explanation": "Uncited", "source_ids": []}
                    ],
                    "open_questions": [],
                    "missing_information": [],
                }
            )
        ]
    )

    response = asyncio.run(
        ResearchBriefService(fake).create_brief(
            "How does this connect?",
            [_source()],
            {"source_1": "Evidence content"},
        )
    )

    assert isinstance(response, ResearchBrief)
    assert fake.chat_calls == 1
    assert response.key_points == []
    assert response.connections == []


def test_search_remains_backward_compatible(tmp_path) -> None:
    settings = _settings_with_database(tmp_path)
    _seed_chunk(settings.database_path)

    response = asyncio.run(
        search_notes(
            SearchRequest(query="planetary habitability", top_k=1),
            settings,
            FakeOllama(),
        )
    )

    assert len(response.results) == 1
    result = response.results[0]
    assert result.filename == "The Long Orbit.md"
    assert result.path == "Projects/The Long Orbit.md"
    assert result.heading == "Methodology"
    assert result.chunk_id == 1
    assert result.score == 1.0
    assert "planetary habitability" in result.content


def _settings_with_database(tmp_path: Path) -> Settings:
    database_path = tmp_path / "atlas.db"
    initialize_database(database_path)
    return Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="qwen3:4b",
        embedding_model="embeddinggemma:latest",
        database_path=database_path,
        obsidian_vault_path=tmp_path,
        default_top_k=5,
        min_top_score=0.35,
        min_average_score=0.25,
        min_meaningful_content_chars=20,
    )


def _seed_chunk(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash", "now"),
        )
        connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cursor.lastrowid,
                "Methodology",
                "The project uses computational modeling for planetary habitability.",
                json.dumps([1.0, 0.0]),
                0,
                "now",
            ),
        )
        connection.commit()


def _seed_long_orbit_chunks(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-long-orbit", "now"),
        )
        file_id = cursor.lastrowid
        rows = [
            (
                "The Long Orbit",
                "# The Long Orbit",
                [1.0, 0.0],
            ),
            (
                "Project Goals",
                "The goal is to model long-term planetary habitability with orbital simulations.",
                [1.0, 0.0],
            ),
            (
                "Checklist",
                "- Complete the simulation\n- Finish deliverables\n- Review completion criteria",
                [0.7, 0.3],
            ),
            (
                "Status",
                "The project is in progress and has unfinished implementation work.",
                [0.7, 0.3],
            ),
        ]
        for index, (heading, content, embedding) in enumerate(rows):
            connection.execute(
                """
                INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_id, heading, content, json.dumps(embedding), index, "now"),
            )
        connection.commit()


def _seed_long_orbit_project_state(database_path: Path) -> None:
    with connect(database_path) as connection:
        file_id = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-entities", "now"),
        ).lastrowid
        chunk_id = connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "Checklist",
                "- [x] Define habitability\n- [ ] Learn stellar evolution basics",
                json.dumps([0.7, 0.3]),
                0,
                "now",
            ),
        ).lastrowid
        project_id = connection.execute(
            """
            INSERT INTO entities(type, name, description, source_file_id, created_at, updated_at)
            VALUES ('Project', 'The Long Orbit', 'Project inferred from The Long Orbit.md', ?, 'now', 'now')
            """,
            (file_id,),
        ).lastrowid
        for key, value in [
            ("display_name", "The Long Orbit"),
            ("status", "In progress."),
            ("phase", "Learning foundations."),
        ]:
            connection.execute(
                """
                INSERT INTO entity_attributes(entity_id, key, value, created_at, updated_at)
                VALUES (?, ?, ?, 'now', 'now')
                """,
                (project_id, key, value),
            )
        for index, (text, state) in enumerate(
            [
                ("Define habitability", "complete"),
                ("Learn stellar evolution basics", "incomplete"),
            ]
        ):
            task_id = connection.execute(
                """
                INSERT INTO entities(type, name, description, source_file_id, created_at, updated_at)
                VALUES ('Task', ?, ?, ?, 'now', 'now')
                """,
                (f"The Long Orbit: {text}", text, file_id),
            ).lastrowid
            for key, value in [
                ("display_name", text),
                ("state", state),
                ("section", "Checklist"),
                ("ordinal", str(index)),
            ]:
                connection.execute(
                    """
                    INSERT INTO entity_attributes(entity_id, key, value, created_at, updated_at)
                    VALUES (?, ?, ?, 'now', 'now')
                    """,
                    (task_id, key, value),
                )
            connection.execute(
                """
                INSERT INTO entity_relationships(
                    source_entity_id,
                    target_entity_id,
                    type,
                    evidence_chunk_id,
                    created_at
                )
                VALUES (?, ?, 'Project -> Task', ?, 'now')
                """,
                (project_id, task_id, chunk_id),
            )
        connection.commit()


def _seed_task_state_chunks(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-tasks", "now"),
        )
        file_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "Checklist",
                "\n".join(
                    [
                        "- [x] Rename `cosmos-lab` to `The Long Orbit`",
                        "- [x] Create `Project Overview.md`",
                        "- [ ] Create a graph of habitable-zone movement",
                        "- Initial model documentation",
                    ]
                ),
                json.dumps([0.7, 0.3]),
                0,
                "now",
            ),
        )
        connection.commit()


def _seed_status_evidence_chunk(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-status", "now"),
        )
        file_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                "Status",
                "No simulation development has started. The graph remains pending.",
                json.dumps([0.7, 0.3]),
                0,
                "now",
            ),
        )
        connection.commit()


def _seed_roadmap_chunks(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-roadmap", "now"),
        )
        file_id = cursor.lastrowid
        rows = [
            ("Learn", "- Stellar luminosity", [0.7, 0.3]),
            ("Deliverables", "- Graph of habitable-zone movement", [0.7, 0.3]),
            ("Completion Criteria", "- Can explain habitable zone migration", [0.7, 0.3]),
            ("Status", "No simulation development has started.", [0.7, 0.3]),
        ]
        for index, (heading, content, embedding) in enumerate(rows):
            connection.execute(
                """
                INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_id, heading, content, json.dumps(embedding), index, "now"),
            )
        connection.commit()


def _seed_mixed_deliverables_chunks(database_path: Path) -> None:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO files(path, filename, content_hash, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Projects/The Long Orbit.md", "The Long Orbit.md", "hash-mixed-deliverables", "now"),
        )
        file_id = cursor.lastrowid
        rows = [
            ("Status", "No simulation development has started.", [0.7, 0.3]),
            (
                "Deliverables",
                "\n".join(
                    [
                        "- Project overview completed",
                        "- Initial journal entry completed",
                        "- Graph of habitable zone movement",
                        "- Initial model documentation",
                    ]
                ),
                [0.7, 0.3],
            ),
        ]
        for index, (heading, content, embedding) in enumerate(rows):
            connection.execute(
                """
                INSERT INTO chunks(file_id, heading, content, embedding, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_id, heading, content, json.dumps(embedding), index, "now"),
            )
        connection.commit()


def _source() -> ResearchSource:
    return ResearchSource(
        id="source_1",
        file="note.md",
        path="note.md",
        heading="Evidence",
        score=0.9,
        excerpt="Evidence content",
    )


def _brief_payload() -> dict[str, object]:
    return {
        "key_points": [{"text": "Evidence point", "source_ids": ["source_1"]}],
        "connections": [
            {
                "concept": "Evidence",
                "explanation": "The note supports this connection.",
                "source_ids": ["source_1"],
            }
        ],
        "open_questions": [],
        "missing_information": [],
    }
