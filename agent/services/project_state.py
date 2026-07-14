from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from enum import StrEnum

from agent.models.schemas import ProjectStateEntity, ProjectStateResponse, ResearchSource


class ProjectStateIntent(StrEnum):
    NEXT_TASK = "next_task"
    UNFINISHED = "unfinished"
    BLOCKED = "blocked"
    PROJECT_SUMMARY = "project_summary"
    PHASE = "phase"
    DECISION = "decision"


NEXT_TASK_PATTERNS = [
    r"\bwhat\s+should\s+i\s+(work\s+on|do|focus\s+on)\s+next\b",
    r"\bnext\s+(task|action|step)\b",
    r"\bhighest\s+priority\s+task\b",
    r"\bwhat\s+should\s+i\s+focus\s+on\b",
]

UNFINISHED_PATTERNS = [
    r"\bwhat\s+remains\s+unfinished\b",
    r"\bwhat\s+remains\s+to\s+be\s+completed\b",
    r"\bwhat\s+tasks?\s+remain\b",
    r"\bwhat\s+is\s+left\s+to\s+do\b",
    r"\bunfinished\b",
]

BLOCKED_PATTERNS = [
    r"\bwhat\s+is\s+blocked\b",
    r"\bwhat\s+am\s+i\s+blocked\s+on\b",
    r"\bblocked\b",
    r"\bblockers?\b",
]

PROJECT_SUMMARY_PATTERNS = [
    r"\bactive\s+projects?\b",
    r"\bwhat\s+projects?\s+are\s+active\b",
    r"\bwhat\s+am\s+i\s+working\s+on\b",
    r"\bsummarize\s+my\s+active\s+projects\b",
    r"\bwhich\s+project\s+needs\s+attention\b",
]

PHASE_PATTERNS = [
    r"\bwhat\s+phase\b",
    r"\bwhich\s+phase\b",
    r"\bcurrent\s+phase\b",
]

DECISION_PATTERNS = [
    r"\bwhat\s+did\s+i\s+decide\b",
    r"\bwhy\s+did\s+i\s+choose\b",
    r"\bwhen\s+was\s+.*decided\b",
    r"\balternatives?\s+.*rejected\b",
]


@dataclass(frozen=True)
class ProjectRecord:
    id: int
    name: str
    status: str | None
    phase: str | None


@dataclass(frozen=True)
class TaskRecord:
    id: int
    text: str
    state: str
    section: str | None
    kind: str
    project: ProjectRecord
    chunk_id: int | None
    file: str | None
    path: str | None
    heading: str | None
    content: str | None
    chunk_index: int
    ordinal: int


def detect_project_state_intent(question: str) -> ProjectStateIntent | None:
    normalized = " ".join(question.lower().split())
    if _matches_any(normalized, NEXT_TASK_PATTERNS):
        return ProjectStateIntent.NEXT_TASK
    if _matches_any(normalized, UNFINISHED_PATTERNS):
        return ProjectStateIntent.UNFINISHED
    if _matches_any(normalized, BLOCKED_PATTERNS):
        return ProjectStateIntent.BLOCKED
    if _matches_any(normalized, PROJECT_SUMMARY_PATTERNS):
        return ProjectStateIntent.PROJECT_SUMMARY
    if _matches_any(normalized, PHASE_PATTERNS):
        return ProjectStateIntent.PHASE
    if _matches_any(normalized, DECISION_PATTERNS):
        return ProjectStateIntent.DECISION
    return None


class ProjectStateService:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def answer(self, question: str, intent: ProjectStateIntent) -> ProjectStateResponse | None:
        projects = self._relevant_projects(question)
        if intent == ProjectStateIntent.PROJECT_SUMMARY:
            return self._project_summary(question, projects)
        if not projects:
            return None
        if intent == ProjectStateIntent.NEXT_TASK:
            return self._next_task(question, projects)
        if intent == ProjectStateIntent.UNFINISHED:
            return self._unfinished(question, projects)
        if intent == ProjectStateIntent.BLOCKED:
            return self._blocked(question, projects)
        if intent == ProjectStateIntent.PHASE:
            return self._phase(question, projects)
        if intent == ProjectStateIntent.DECISION:
            return self._decisions(question, projects)
        return None

    def _project_summary(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse | None:
        projects = [project for project in projects if _is_active(project)]
        if not projects:
            return None
        entities = [
            ProjectStateEntity(
                id=project.id,
                type="Project",
                name=project.name,
                state=project.status,
            )
            for project in projects
        ]
        fragments = []
        missing: list[str] = []
        for project in projects:
            status = project.status or "status unknown"
            phase = effective_phase(project) or "phase unknown"
            fragments.append(f"{project.name}: {status}; {phase}")
            if project.status is None:
                missing.append(f"{project.name} has no indexed status.")
            if effective_phase(project) is None:
                missing.append(f"{project.name} has no indexed phase.")
        return ProjectStateResponse(
            question=question,
            answer="Active projects: " + "; ".join(fragments),
            recommended_action=None,
            entities=entities,
            missing_information=missing,
            sources=[],
        )

    def _next_task(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse | None:
        tasks = self._tasks_for_projects(projects)
        unfinished = [task for task in tasks if _is_incomplete_task(task)]
        unclear = [task for task in tasks if _is_unknown_task(task)]
        candidate = (unfinished or unclear or [None])[0]
        if candidate is None:
            return ProjectStateResponse(
                question=question,
                answer=f"No unfinished tasks are indexed for {_project_names(projects)}.",
                recommended_action=None,
                entities=[
                    ProjectStateEntity(id=project.id, type="Project", name=project.name, state=project.status)
                    for project in projects
                ],
                missing_information=["Atlas found no incomplete or unclear task entities for the selected project."],
                sources=[],
            )
        missing = _project_missing(candidate.project)
        if not unfinished and unclear:
            missing.append("The recommended item has unclear completion status because it was not marked with a checkbox.")
        answer = (
            f"Work on this next for {candidate.project.name}: {candidate.text}. "
            f"Current phase: {effective_phase(candidate.project) or 'unknown'}."
        )
        return ProjectStateResponse(
            question=question,
            answer=answer,
            recommended_action=candidate.text,
            entities=[
                ProjectStateEntity(
                    id=candidate.project.id,
                    type="Project",
                    name=candidate.project.name,
                    state=candidate.project.status,
                ),
                ProjectStateEntity(
                    id=candidate.id,
                    type="Task",
                    name=candidate.text,
                    state=candidate.state,
                    project=candidate.project.name,
                ),
            ],
            missing_information=missing,
            sources=_sources_for_tasks([candidate]),
        )

    def _unfinished(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse | None:
        tasks = self._tasks_for_projects(projects)
        unfinished = [task for task in tasks if _is_incomplete_task(task)]
        unclear = [task for task in tasks if _is_unknown_task(task)]
        relevant = unfinished + unclear
        if not relevant:
            return ProjectStateResponse(
                question=question,
                answer=f"No unfinished tasks are indexed for {_project_names(projects)}.",
                recommended_action=None,
                entities=[
                    ProjectStateEntity(id=project.id, type="Project", name=project.name, state=project.status)
                    for project in projects
                ],
                missing_information=["Atlas found no incomplete or unclear task entities for the selected project."],
                sources=[],
            )
        task_text = "; ".join(f"{task.text} ({task.state})" for task in relevant)
        recommended = relevant[0].text
        return ProjectStateResponse(
            question=question,
            answer=f"Unfinished work: {task_text}.",
            recommended_action=recommended,
            entities=[
                ProjectStateEntity(
                    id=task.id,
                    type="Task",
                    name=task.text,
                    state=task.state,
                    project=task.project.name,
                )
                for task in relevant
            ],
            missing_information=[
                "Plain bullet items have unclear completion status."
            ]
            if unclear
            else [],
            sources=_sources_for_tasks(relevant),
        )

    def _phase(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse:
        missing = []
        fragments = []
        for project in projects:
            phase = effective_phase(project)
            if phase:
                fragments.append(f"{project.name} is in phase: {phase}")
            else:
                fragments.append(f"{project.name} has no indexed phase")
                missing.append(f"{project.name} has no indexed phase.")
        return ProjectStateResponse(
            question=question,
            answer="; ".join(fragments),
            recommended_action=None,
            entities=[
                ProjectStateEntity(id=project.id, type="Project", name=project.name, state=project.status)
                for project in projects
            ],
            missing_information=missing,
            sources=[],
        )

    def _blocked(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse:
        blockers = [
            task
            for task in self._tasks_for_projects(projects)
            if task.kind == "blocker" and not _is_completed_task(task)
        ]
        if not blockers:
            return ProjectStateResponse(
                question=question,
                answer=f"No blockers are indexed for {_project_names(projects)}.",
                recommended_action=None,
                entities=[
                    ProjectStateEntity(id=project.id, type="Project", name=project.name, state=project.status)
                    for project in projects
                ],
                missing_information=["Blockers are only available when a note has a Blockers section or blocker task entries."],
                sources=[],
            )
        return ProjectStateResponse(
            question=question,
            answer="Indexed blockers: " + "; ".join(f"{task.project.name}: {task.text}" for task in blockers),
            recommended_action=blockers[0].text,
            entities=[
                ProjectStateEntity(
                    id=task.id,
                    type="Task",
                    name=task.text,
                    state=task.state,
                    project=task.project.name,
                )
                for task in blockers
            ],
            missing_information=[],
            sources=_sources_for_tasks(blockers),
        )

    def _decisions(
        self,
        question: str,
        projects: list[ProjectRecord],
    ) -> ProjectStateResponse | None:
        project_ids = [project.id for project in projects]
        placeholders = ", ".join("?" for _ in project_ids)
        rows = self.connection.execute(
            f"""
            SELECT decisions.id, decisions.description, projects.name AS project_name,
                   chunks.id AS chunk_id, chunks.heading, chunks.content, files.filename, files.path
            FROM entity_relationships AS relationships
            JOIN entities AS decisions ON decisions.id = relationships.source_entity_id
            JOIN entities AS projects ON projects.id = relationships.target_entity_id
            LEFT JOIN chunks ON chunks.id = relationships.evidence_chunk_id
            LEFT JOIN files ON files.id = chunks.file_id
            WHERE relationships.type = 'Decision -> Project'
              AND projects.id IN ({placeholders})
            ORDER BY decisions.id
            """,
            project_ids,
        ).fetchall()
        if not rows:
            return ProjectStateResponse(
                question=question,
                answer=f"No decision records are indexed for {_project_names(projects)}.",
                recommended_action=None,
                entities=[
                    ProjectStateEntity(id=project.id, type="Project", name=project.name, state=project.status)
                    for project in projects
                ],
                missing_information=["Atlas found no Decision entities for the selected project."],
                sources=[],
            )
        decisions = []
        missing = []
        for row in rows:
            attrs = self._attributes(int(row["id"]))
            parts = [row["description"]]
            if attrs.get("date"):
                parts.append(f"date: {attrs['date']}")
            else:
                missing.append(f"Decision '{row['description']}' has no indexed date.")
            if attrs.get("reasoning"):
                parts.append(f"reasoning: {attrs['reasoning']}")
            else:
                missing.append(f"Decision '{row['description']}' has no indexed reasoning.")
            if attrs.get("alternatives"):
                parts.append(f"alternatives: {attrs['alternatives']}")
            else:
                missing.append(f"Decision '{row['description']}' has no indexed alternatives.")
            decisions.append(" | ".join(parts))
        return ProjectStateResponse(
            question=question,
            answer="Indexed decisions: " + "; ".join(decisions),
            recommended_action=None,
            entities=[
                ProjectStateEntity(
                    id=int(row["id"]),
                    type="Decision",
                    name=row["description"],
                    project=row["project_name"],
                )
                for row in rows
            ],
            missing_information=missing,
            sources=_sources_from_rows(rows),
        )

    def _relevant_projects(self, question: str) -> list[ProjectRecord]:
        projects = self._projects()
        normalized_question = question.lower()
        matching = [project for project in projects if project.name.lower() in normalized_question]
        return matching or projects

    def _projects(self) -> list[ProjectRecord]:
        rows = self.connection.execute(
            """
            SELECT id, name
            FROM entities
            WHERE type = 'Project'
            ORDER BY updated_at DESC, name
            """
        ).fetchall()
        projects: list[ProjectRecord] = []
        for row in rows:
            attrs = self._attributes(int(row["id"]))
            projects.append(
                ProjectRecord(
                    id=int(row["id"]),
                    name=attrs.get("display_name", row["name"]),
                    status=attrs.get("status"),
                    phase=attrs.get("phase"),
                )
            )
        return projects

    def _tasks_for_projects(self, projects: list[ProjectRecord]) -> list[TaskRecord]:
        if not projects:
            return []
        project_by_id = {project.id: project for project in projects}
        placeholders = ", ".join("?" for _ in project_by_id)
        rows = self.connection.execute(
            f"""
            SELECT projects.id AS project_id, tasks.id AS task_id, tasks.description,
                   relationships.evidence_chunk_id, chunks.heading, chunks.content,
                   chunks.chunk_index, files.filename, files.path
            FROM entity_relationships AS relationships
            JOIN entities AS projects ON projects.id = relationships.source_entity_id
            JOIN entities AS tasks ON tasks.id = relationships.target_entity_id
            LEFT JOIN chunks ON chunks.id = relationships.evidence_chunk_id
            LEFT JOIN files ON files.id = chunks.file_id
            WHERE relationships.type = 'Project -> Task'
              AND projects.id IN ({placeholders})
            """,
            list(project_by_id),
        ).fetchall()
        tasks: list[TaskRecord] = []
        for row in rows:
            attrs = self._attributes(int(row["task_id"]))
            tasks.append(
                TaskRecord(
                    id=int(row["task_id"]),
                    text=attrs.get("display_name", row["description"]),
                    state=attrs.get("state", "unknown"),
                    section=attrs.get("section"),
                    kind=attrs.get("kind", "task"),
                    project=project_by_id[int(row["project_id"])],
                    chunk_id=int(row["evidence_chunk_id"]) if row["evidence_chunk_id"] is not None else None,
                    file=row["filename"],
                    path=row["path"],
                    heading=row["heading"],
                    content=row["content"],
                    chunk_index=int(row["chunk_index"]) if row["chunk_index"] is not None else 9999,
                    ordinal=int(attrs.get("ordinal", "9999")),
                )
            )
        tasks.sort(key=lambda task: (task.chunk_index, task.ordinal, task.text.lower()))
        return tasks

    def _attributes(self, entity_id: int) -> dict[str, str]:
        rows = self.connection.execute(
            "SELECT key, value FROM entity_attributes WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}


def _sources_for_tasks(tasks: list[TaskRecord]) -> list[ResearchSource]:
    rows = []
    for task in tasks:
        if task.chunk_id is None:
            continue
        rows.append(
            {
                "chunk_id": task.chunk_id,
                "filename": task.file,
                "path": task.path,
                "heading": task.heading,
                "content": task.content,
            }
        )
    return _sources_from_rows(rows)


def _sources_from_rows(rows: list[sqlite3.Row] | list[dict[str, object]]) -> list[ResearchSource]:
    sources: list[ResearchSource] = []
    seen: set[int] = set()
    for row in rows:
        chunk_id = row["chunk_id"]
        if chunk_id is None or int(chunk_id) in seen:
            continue
        seen.add(int(chunk_id))
        content = str(row["content"] or "")
        sources.append(
            ResearchSource(
                id=f"source_{len(sources) + 1}",
                file=str(row["filename"] or ""),
                path=str(row["path"] or ""),
                heading=str(row["heading"] or ""),
                score=1.0,
                excerpt=_excerpt(content),
            )
        )
    return sources


def _excerpt(content: str, limit: int = 220) -> str:
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _project_missing(project: ProjectRecord) -> list[str]:
    missing = []
    if effective_phase(project) is None:
        missing.append(f"{project.name} has no indexed phase.")
    if project.status is None:
        missing.append(f"{project.name} has no indexed status.")
    return missing


def effective_phase(project: ProjectRecord) -> str | None:
    if project.phase:
        return project.phase
    if project.status and _looks_like_phase(project.status):
        return project.status
    return None


def _project_names(projects: list[ProjectRecord]) -> str:
    return ", ".join(project.name for project in projects)


def _is_incomplete_task(task: TaskRecord) -> bool:
    return task.state == "incomplete"


def _is_unknown_task(task: TaskRecord) -> bool:
    return task.state == "unknown"


def _is_completed_task(task: TaskRecord) -> bool:
    return task.state in {"complete", "completed"}


def _is_active(project: ProjectRecord) -> bool:
    if project.status is None:
        return True
    return not re.search(r"\b(done|complete|completed|inactive|archived|paused)\b", project.status, re.IGNORECASE)


def _looks_like_phase(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"active", "paused", "completed", "archived", "blocked", "done", "inactive"}:
        return False
    return bool(re.search(r"\b(phase|stage)\b", normalized))


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
