from __future__ import annotations

import re
from dataclasses import replace
from enum import StrEnum

from agent.models.schemas import RetrievalAssessment
from agent.retrieval.search import RetrievedChunk


class QueryIntent(StrEnum):
    RESEARCH = "research"
    STATUS = "status"
    COMPLETION = "completion"


class HeadingKind(StrEnum):
    STATUS = "status"
    COMPLETION_CRITERIA = "completion_criteria"
    DELIVERABLES = "deliverables"
    ACTIVITIES = "activities"
    GOAL = "goal"
    LOW_VALUE_COMPLETION = "low_value_completion"
    OTHER = "other"


ADMINISTRATIVE_HEADINGS = {
    "checklist",
    "completion criteria",
    "deliverables",
    "status",
    "todo",
    "to do",
    "tasks",
    "roadmap",
    "milestones",
}

STATUS_HEADINGS = {"status", "current status", "progress", "phase"}
COMPLETION_CRITERIA_HEADINGS = {"completion criteria", "success criteria", "done when"}
DELIVERABLE_HEADINGS = {"deliverables", "outputs", "artifacts"}
ACTIVITY_HEADINGS = {"activities", "tasks", "next steps", "todo", "to do"}
GOAL_HEADINGS = {"goal", "objective"}
LOW_VALUE_COMPLETION_HEADINGS = {
    "learn",
    "notes",
    "reading targets",
    "references",
    "resources",
    "topics",
}

COMPLETION_HEADING_BOOSTS = {
    HeadingKind.STATUS: 3.0,
    HeadingKind.COMPLETION_CRITERIA: 2.5,
    HeadingKind.DELIVERABLES: 2.0,
    HeadingKind.ACTIVITIES: 1.5,
    HeadingKind.GOAL: 1.2,
}

COMPLETION_HEADING_PENALTIES = {
    HeadingKind.LOW_VALUE_COMPLETION: 0.5,
}

COMPLETION_CATEGORY_CAPS = {
    HeadingKind.STATUS: 1,
    HeadingKind.COMPLETION_CRITERIA: 2,
    HeadingKind.DELIVERABLES: 2,
    HeadingKind.ACTIVITIES: 1,
    HeadingKind.GOAL: 1,
    HeadingKind.OTHER: 1,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "completed",
    "does",
    "explain",
    "for",
    "how",
    "in",
    "is",
    "it",
    "my",
    "of",
    "remains",
    "say",
    "the",
    "to",
    "what",
}

COMPLETION_PATTERNS = [
    r"\bwhat\s+remains\b",
    r"\bremaining\b",
    r"\bto\s+do\b",
    r"\btodo\b",
    r"\bchecklist\b",
    r"\bcompletion\b",
    r"\bdeliverables?\b",
    r"\bunfinished\b",
    r"\bleft\s+to\s+do\b",
]

STATUS_PATTERNS = [
    r"\bprogress\b",
    r"\bstatus\b",
    r"\bwhere\s+.*\bstand\b",
]

RESEARCH_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bexplain\b",
    r"\bhow\s+does\b",
    r"\brelationship\b",
    r"\bevidence\b",
    r"\bresearch\b",
]


def detect_query_intent(question: str) -> QueryIntent:
    normalized = " ".join(question.lower().split())
    if _matches_any(normalized, COMPLETION_PATTERNS):
        return QueryIntent.COMPLETION
    if _matches_any(normalized, STATUS_PATTERNS):
        return QueryIntent.STATUS
    return QueryIntent.RESEARCH


def rank_for_research(
    chunks: list[RetrievedChunk],
    intent: QueryIntent,
    min_meaningful_content_chars: int,
) -> list[RetrievedChunk]:
    candidates = [
        chunk
        for chunk in chunks
        if has_meaningful_content(chunk, min_meaningful_content_chars)
    ]
    ranked = [_adjusted_chunk(chunk, intent) for chunk in candidates]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def select_diverse_completion_chunks(
    chunks: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    if limit <= 0:
        return []

    selected: list[RetrievedChunk] = []
    selected_ids: set[int] = set()
    category_counts: dict[HeadingKind, int] = {}
    priority = [
        HeadingKind.STATUS,
        HeadingKind.COMPLETION_CRITERIA,
        HeadingKind.DELIVERABLES,
        HeadingKind.ACTIVITIES,
    ]

    for heading_kind in priority:
        candidate = _best_chunk_for_kind(chunks, heading_kind, selected_ids)
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)
            _increment_category_count(category_counts, candidate)
        if len(selected) >= limit:
            return selected

    for chunk in chunks:
        if chunk.chunk_id in selected_ids:
            continue
        if not _within_completion_category_cap(chunk, category_counts):
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        _increment_category_count(category_counts, chunk)
        if len(selected) >= limit:
            break
    if len(selected) >= limit:
        return selected
    for chunk in chunks:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        if len(selected) >= limit:
            break
    return selected


def assess_retrieval_quality(
    question: str,
    chunks: list[RetrievedChunk],
    min_top_score: float,
    min_average_score: float,
) -> RetrievalAssessment:
    if not chunks:
        return RetrievalAssessment(
            supported=False,
            confidence=0.0,
            reason="insufficient_evidence",
        )

    top_score = chunks[0].score
    average_score = sum(chunk.score for chunk in chunks) / len(chunks)
    confidence = max(0.0, min(1.0, (top_score + average_score) / 2))
    if not _has_evidence_overlap(question, chunks) and top_score < 0.72:
        return RetrievalAssessment(
            supported=False,
            confidence=confidence,
            reason="insufficient_relevance",
        )

    if top_score < min_top_score:
        return RetrievalAssessment(
            supported=False,
            confidence=confidence,
            reason="insufficient_relevance",
        )
    if average_score < min_average_score:
        return RetrievalAssessment(
            supported=False,
            confidence=confidence,
            reason="insufficient_evidence",
        )
    return RetrievalAssessment(
        supported=True,
        confidence=confidence,
        reason="supported",
    )


def has_meaningful_content(chunk: RetrievedChunk, min_chars: int) -> bool:
    meaningful = _meaningful_content(chunk)
    if len(meaningful) >= min_chars:
        return True
    return _looks_like_definition(meaningful)


def is_administrative_heading(heading: str) -> bool:
    normalized = _normalize_heading(heading)
    return normalized in ADMINISTRATIVE_HEADINGS


def classify_heading(heading: str) -> HeadingKind:
    normalized = _normalize_heading(heading)
    if normalized in STATUS_HEADINGS:
        return HeadingKind.STATUS
    if normalized in COMPLETION_CRITERIA_HEADINGS:
        return HeadingKind.COMPLETION_CRITERIA
    if normalized in DELIVERABLE_HEADINGS:
        return HeadingKind.DELIVERABLES
    if normalized in ACTIVITY_HEADINGS:
        return HeadingKind.ACTIVITIES
    if normalized in GOAL_HEADINGS:
        return HeadingKind.GOAL
    if normalized in LOW_VALUE_COMPLETION_HEADINGS:
        return HeadingKind.LOW_VALUE_COMPLETION
    return HeadingKind.OTHER


def _adjusted_chunk(chunk: RetrievedChunk, intent: QueryIntent) -> RetrievedChunk:
    administrative = is_administrative_heading(chunk.heading)
    heading_kind = classify_heading(chunk.heading)
    adjustment = 0.0
    if intent == QueryIntent.COMPLETION:
        adjustment = COMPLETION_HEADING_BOOSTS.get(heading_kind, 0.0)
        adjustment -= COMPLETION_HEADING_PENALTIES.get(heading_kind, 0.0)
    elif intent == QueryIntent.STATUS:
        adjustment = {
            HeadingKind.STATUS: 3.0,
            HeadingKind.COMPLETION_CRITERIA: 2.0,
            HeadingKind.DELIVERABLES: 1.5,
            HeadingKind.ACTIVITIES: 1.2,
            HeadingKind.GOAL: 0.8,
            HeadingKind.LOW_VALUE_COMPLETION: -0.5,
        }.get(heading_kind, 0.0)
    elif administrative and intent == QueryIntent.RESEARCH:
        adjustment = -0.08
    return replace(chunk, score=chunk.score + adjustment)


def _meaningful_content(chunk: RetrievedChunk) -> str:
    lines = []
    for line in chunk.content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lstrip("#").strip().lower() == chunk.heading.lower():
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return " ".join(lines).strip()


def _looks_like_definition(content: str) -> bool:
    if len(content) < 12:
        return False
    return bool(re.search(r"\b(is|are|means|refers to|defined as)\b", content, re.IGNORECASE))


def _normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.strip().lower().replace("-", " "))


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _best_chunk_for_kind(
    chunks: list[RetrievedChunk],
    heading_kind: HeadingKind,
    selected_ids: set[int],
) -> RetrievedChunk | None:
    for chunk in chunks:
        if chunk.chunk_id in selected_ids:
            continue
        if classify_heading(chunk.heading) == heading_kind:
            return chunk
    return None


def _within_completion_category_cap(
    chunk: RetrievedChunk,
    category_counts: dict[HeadingKind, int],
) -> bool:
    heading_kind = classify_heading(chunk.heading)
    cap = COMPLETION_CATEGORY_CAPS.get(heading_kind)
    if cap is None:
        return True
    return category_counts.get(heading_kind, 0) < cap


def _increment_category_count(
    category_counts: dict[HeadingKind, int],
    chunk: RetrievedChunk,
) -> None:
    heading_kind = classify_heading(chunk.heading)
    category_counts[heading_kind] = category_counts.get(heading_kind, 0) + 1


def _has_evidence_overlap(question: str, chunks: list[RetrievedChunk]) -> bool:
    terms = _keywords(question)
    if not terms:
        return True
    evidence_text = " ".join(
        f"{chunk.filename} {chunk.path} {chunk.heading} {chunk.content}"
        for chunk in chunks
    ).lower()
    evidence_terms = set(re.findall(r"[a-z0-9]+", evidence_text))
    return bool(terms & evidence_terms)


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }
