from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.services.task_parser import parse_task_items


PROJECT_HEADINGS = {"project", "overview", "project overview"}
STATUS_HEADINGS = {"status", "current status", "progress"}
PHASE_HEADINGS = {"current phase", "current stage", "phase", "project phase", "stage"}
GOAL_HEADINGS = {"goal", "goals", "project goals", "objective", "objectives"}
DELIVERABLE_HEADINGS = {"deliverable", "deliverables", "outputs", "artifacts"}
TASK_HEADINGS = {
    "action items",
    "checklist",
    "current phase checklist",
    "next steps",
    "task",
    "tasks",
    "to do",
    "todo",
}
MILESTONE_HEADINGS = {"milestone", "milestones"}
BLOCKER_HEADINGS = {"blocker", "blockers", "blocked"}
DECISION_HEADINGS = {"decision", "decisions", "decision log", "architecture decisions"}
CONCEPT_HEADINGS = {"concept", "concepts", "key concepts"}
RESEARCH_TOPIC_HEADINGS = {"research topic", "research topics", "questions", "open questions"}
REPOSITORY_HEADINGS = {"repository", "repositories", "repo", "repos"}
JOURNAL_HEADINGS = {"journal", "journal entry", "log", "work log"}


@dataclass(frozen=True)
class IndexedChunk:
    id: int
    heading: str
    content: str
    chunk_index: int


def extract_entities_for_file(
    connection: sqlite3.Connection,
    *,
    file_id: int,
    path: Path,
    chunks: list[IndexedChunk],
) -> None:
    project_name = _project_name(connection, path, chunks)
    if project_name is None:
        return
    now = datetime.now(timezone.utc).isoformat()

    project_id = _upsert_entity(
        connection,
        entity_type="Project",
        name=project_name,
        description=f"Project inferred from {path.name}",
        source_file_id=file_id,
        now=now,
    )
    _set_attribute(connection, project_id, "display_name", project_name, now)
    _set_attribute(connection, project_id, "source_path", str(path), now)

    for chunk in chunks:
        heading = _normalize_heading(chunk.heading)
        content_body = _content_without_heading(chunk.content)

        if heading in STATUS_HEADINGS:
            status = _first_meaningful_line(content_body)
            if status:
                _set_attribute(connection, project_id, "status", status, now)

        if heading in PHASE_HEADINGS:
            phase = _first_meaningful_line(content_body)
            if phase:
                _set_attribute(connection, project_id, "phase", phase, now)

        if heading in GOAL_HEADINGS or heading.endswith(" goal") or heading.endswith(" goals"):
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Goal",
                relationship_type="Project -> Goal",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
            )

        if (
            heading in DELIVERABLE_HEADINGS
            or heading.endswith(" deliverable")
            or heading.endswith(" deliverables")
        ):
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Deliverable",
                relationship_type="Project -> Deliverable",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
            )

        if (
            heading in TASK_HEADINGS
            or heading in MILESTONE_HEADINGS
            or heading in BLOCKER_HEADINGS
            or _has_checkboxes(content_body)
        ):
            if heading in MILESTONE_HEADINGS:
                kind = "milestone"
            elif heading in BLOCKER_HEADINGS:
                kind = "blocker"
            else:
                kind = "task"
            _extract_tasks(
                connection,
                project_id=project_id,
                project_name=project_name,
                section=chunk.heading,
                kind=kind,
                content=content_body,
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
            )

        if heading in DECISION_HEADINGS:
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Decision",
                relationship_type="Decision -> Project",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
                reverse_relationship=True,
            )

        if heading in CONCEPT_HEADINGS:
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Concept",
                relationship_type="Project -> Concept",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
            )

        if heading in RESEARCH_TOPIC_HEADINGS:
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Research Topic",
                relationship_type="Project -> Research Topic",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
            )

        if heading in REPOSITORY_HEADINGS:
            _extract_list_entities(
                connection,
                project_id=project_id,
                project_name=project_name,
                entity_type="Repository",
                relationship_type="Repository -> Project",
                texts=_bullet_or_lines(content_body),
                file_id=file_id,
                chunk_id=chunk.id,
                now=now,
                reverse_relationship=True,
            )

        if heading in JOURNAL_HEADINGS:
            entry_id = _upsert_entity(
                connection,
                entity_type="Journal Entry",
                name=f"{project_name}: {path.stem}",
                description=_first_meaningful_line(content_body) or path.stem,
                source_file_id=file_id,
                now=now,
            )
            _set_attribute(connection, entry_id, "display_name", path.stem, now)
            _insert_relationship(
                connection,
                source_entity_id=entry_id,
                target_entity_id=project_id,
                relationship_type="Journal Entry -> Project",
                chunk_id=chunk.id,
                now=now,
            )


def delete_entities_for_file(connection: sqlite3.Connection, file_id: int) -> None:
    connection.execute("DELETE FROM entities WHERE source_file_id = ?", (file_id,))


def _extract_tasks(
    connection: sqlite3.Connection,
    *,
    project_id: int,
    project_name: str,
    section: str,
    kind: str,
    content: str,
    file_id: int,
    chunk_id: int,
    now: str,
) -> None:
    for index, task in enumerate(parse_task_items(content)):
        task_id = _upsert_entity(
            connection,
            entity_type="Task",
            name=f"{project_name}: {task.text}",
            description=task.text,
            source_file_id=file_id,
            now=now,
        )
        _set_attribute(connection, task_id, "display_name", task.text, now)
        # Plain bullets in task-oriented sections are actionable unless explicit checkbox state says otherwise.
        _set_attribute(connection, task_id, "state", _graph_task_state(task.state), now)
        _set_attribute(connection, task_id, "section", section, now)
        _set_attribute(connection, task_id, "kind", kind, now)
        _set_attribute(connection, task_id, "ordinal", str(index), now)
        _insert_relationship(
            connection,
            source_entity_id=project_id,
            target_entity_id=task_id,
            relationship_type="Project -> Task",
            chunk_id=chunk_id,
            now=now,
        )


def _extract_list_entities(
    connection: sqlite3.Connection,
    *,
    project_id: int,
    project_name: str,
    entity_type: str,
    relationship_type: str,
    texts: list[str],
    file_id: int,
    chunk_id: int,
    now: str,
    reverse_relationship: bool = False,
) -> None:
    for index, text in enumerate(texts):
        if not text:
            continue
        entity_id = _upsert_entity(
            connection,
            entity_type=entity_type,
            name=f"{project_name}: {text}",
            description=text,
            source_file_id=file_id,
            now=now,
        )
        _set_attribute(connection, entity_id, "display_name", text, now)
        _set_attribute(connection, entity_id, "ordinal", str(index), now)
        if entity_type == "Decision":
            for key, value in _decision_attributes(text).items():
                _set_attribute(connection, entity_id, key, value, now)
        if reverse_relationship:
            source_entity_id = entity_id
            target_entity_id = project_id
        else:
            source_entity_id = project_id
            target_entity_id = entity_id
        _insert_relationship(
            connection,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            chunk_id=chunk_id,
            now=now,
        )


def _upsert_entity(
    connection: sqlite3.Connection,
    *,
    entity_type: str,
    name: str,
    description: str | None,
    source_file_id: int | None,
    now: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO entities(type, name, description, source_file_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(type, name) DO UPDATE SET
            description = excluded.description,
            source_file_id = COALESCE(excluded.source_file_id, entities.source_file_id),
            updated_at = excluded.updated_at
        RETURNING id
        """,
        (entity_type, name, description, source_file_id, now, now),
    )
    return int(cursor.fetchone()["id"])


def _set_attribute(
    connection: sqlite3.Connection,
    entity_id: int,
    key: str,
    value: str,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT INTO entity_attributes(entity_id, key, value, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entity_id, key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (entity_id, key, value, now, now),
    )


def _insert_relationship(
    connection: sqlite3.Connection,
    *,
    source_entity_id: int,
    target_entity_id: int,
    relationship_type: str,
    chunk_id: int,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO entity_relationships(
            source_entity_id,
            target_entity_id,
            type,
            evidence_chunk_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_entity_id, target_entity_id, relationship_type, chunk_id, now),
    )


def _project_name(
    connection: sqlite3.Connection,
    path: Path,
    chunks: list[IndexedChunk],
) -> str | None:
    project_from_path = _project_name_from_path(path)
    if project_from_path is not None:
        return project_from_path

    referenced_project = _referenced_existing_project(connection, chunks)
    if referenced_project is not None:
        return referenced_project

    for chunk in chunks:
        for line in chunk.content.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*$", line)
            if not match:
                continue
            heading = match.group(1).strip()
            if _looks_like_phase_heading(heading) or _is_inbox_path(path):
                return None
            return heading
    return path.stem


def _project_name_from_path(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        normalized = _normalize_heading(part)
        if normalized == "projects" or re.fullmatch(r"\d+\s+projects", normalized):
            next_part = parts[index + 1]
            if next_part.endswith(".md"):
                return Path(next_part).stem
            return next_part
    return None


def _referenced_existing_project(
    connection: sqlite3.Connection,
    chunks: list[IndexedChunk],
) -> str | None:
    rows = connection.execute(
        "SELECT name FROM entities WHERE type = 'Project' ORDER BY length(name) DESC"
    ).fetchall()
    content = "\n".join(chunk.content for chunk in chunks)
    matches = [
        row["name"]
        for row in rows
        if _has_explicit_project_reference(content, str(row["name"]))
    ]
    if len(matches) == 1:
        return str(matches[0])
    return None


def _has_explicit_project_reference(content: str, project_name: str) -> bool:
    escaped = re.escape(project_name)
    return bool(
        re.search(rf"(?im)^\s*project\s*:\s*{escaped}\s*$", content)
        or re.search(rf"\[\[{escaped}(?:\|[^\]]+)?\]\]", content)
    )


def _looks_like_phase_heading(heading: str) -> bool:
    return bool(re.match(r"phase\s+\d+\s*:", heading.strip(), re.IGNORECASE))


def _is_inbox_path(path: Path) -> bool:
    return any(_normalize_heading(part) == "00 inbox" for part in path.parts)


def _content_without_heading(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        return "\n".join(lines[1:]).strip()
    return content.strip()


def _bullet_or_lines(content: str) -> list[str]:
    items: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^[-*+]\s+(?:\[[ xX]\]\s+)?(.+?)\s*$", stripped)
        if match:
            items.append(match.group(1).strip())
        elif not stripped.startswith("#"):
            items.append(stripped)
    return items


def _first_meaningful_line(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return re.sub(r"^[-*+]\s+", "", stripped)
    return None


def _has_checkboxes(content: str) -> bool:
    return bool(re.search(r"^\s*[-*+]\s+\[[ xX]\]\s+", content, re.MULTILINE))


def _normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.strip().lower().replace("-", " "))


def _graph_task_state(state: str) -> str:
    if state == "complete":
        return "completed"
    return state


def _decision_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if date_match:
        attrs["date"] = date_match.group(1)
    reasoning_match = re.search(r"\bbecause\b\s+(.+?)(?:\bAlternatives?:|\bRejected:|$)", text, re.IGNORECASE)
    if reasoning_match:
        attrs["reasoning"] = reasoning_match.group(1).strip(" .;")
    alternatives_match = re.search(r"\bAlternatives?:\s*(.+)$", text, re.IGNORECASE)
    if alternatives_match:
        attrs["alternatives"] = alternatives_match.group(1).strip(" .;")
    rejected_match = re.search(r"\bRejected:\s*(.+)$", text, re.IGNORECASE)
    if rejected_match:
        attrs["alternatives"] = rejected_match.group(1).strip(" .;")
    return attrs
