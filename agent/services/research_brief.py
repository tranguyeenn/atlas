from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from agent.models.schemas import (
    LLMResearchBrief,
    ResearchBrief,
    ResearchConnection,
    ResearchErrorResponse,
    ResearchKeyPoint,
    ResearchSource,
    TaskItem,
)
from agent.services.ollama import OllamaClient
from agent.services.retrieval_quality import HeadingKind, QueryIntent, classify_heading
from agent.services.task_parser import has_explicit_incomplete_evidence, parse_task_items


CONTROLLED_FAILURE_MESSAGE = (
    "Atlas could not produce a valid cited research brief from the retrieved notes."
)
NO_THINK_PREFIX = "/no_think\n"
logger = logging.getLogger("atlas")


class ResearchBriefService:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    async def create_brief(
        self,
        question: str,
        sources: list[ResearchSource],
        source_contents: dict[str, str] | None = None,
        intent: QueryIntent = QueryIntent.RESEARCH,
    ) -> ResearchBrief | ResearchErrorResponse:
        task_items = _tasks_by_source(source_contents or {})
        messages = _brief_messages(question, sources, source_contents or {}, intent, task_items)
        generation_started_at = perf_counter()
        raw_response = await self.ollama.chat(messages, json_mode=True)
        logger.info("research generation_seconds=%.3f", perf_counter() - generation_started_at)

        parsed_result = parse_llm_research_brief_detailed(raw_response)
        logger.info("research initial_parse_success=%s", str(parsed_result.brief is not None).lower())
        logger.info(
            "research deterministic_cleanup_used=%s",
            str(parsed_result.cleanup_used).lower(),
        )
        if parsed_result.cleanup_reason:
            logger.info("research deterministic_cleanup_reason=%s", parsed_result.cleanup_reason)
        if parsed_result.brief is None:
            _log_validation_failure("initial", parsed_result)

        parsed = parsed_result.brief
        error = parsed_result.error_message
        if parsed is None:
            repair_started_at = perf_counter()
            repair_response = await self.ollama.chat(
                _repair_messages(sources, raw_response, error),
                json_mode=True,
            )
            logger.info("research repair_attempted=true")
            logger.info("research repair_seconds=%.3f", perf_counter() - repair_started_at)
            repair_result = parse_llm_research_brief_detailed(repair_response)
            if repair_result.brief is None:
                _log_validation_failure("repair", repair_result)
            parsed = repair_result.brief
        else:
            logger.info("research repair_attempted=false")

        if parsed is None:
            return ResearchErrorResponse(message=CONTROLLED_FAILURE_MESSAGE)

        sanitized = validate_source_references(parsed, sources)
        if intent in {QueryIntent.COMPLETION, QueryIntent.STATUS}:
            sanitized = _apply_completion_state_awareness(
                sanitized,
                sources,
                source_contents or {},
                task_items,
            )
        return ResearchBrief(
            question=question,
            key_points=sanitized.key_points,
            connections=sanitized.connections,
            open_questions=sanitized.open_questions,
            missing_information=sanitized.missing_information,
            sources=sources,
        )


@dataclass(frozen=True)
class ParseResult:
    brief: LLMResearchBrief | None
    error_message: str | None
    reason: str | None
    field: str | None = None
    cleanup_used: bool = False
    cleanup_reason: str | None = None
    raw_preview: str | None = None


@dataclass(frozen=True)
class SanitizedResearchBrief:
    key_points: list[ResearchKeyPoint]
    connections: list[ResearchConnection]
    open_questions: list[str]
    missing_information: list[str]


def parse_llm_research_brief(raw_response: str) -> tuple[LLMResearchBrief | None, str | None]:
    result = parse_llm_research_brief_detailed(raw_response)
    return result.brief, result.error_message


def parse_llm_research_brief_detailed(raw_response: str) -> ParseResult:
    if raw_response == "":
        return ParseResult(
            brief=None,
            error_message="Response was empty.",
            reason="empty_response",
            raw_preview=_debug_preview(raw_response),
        )

    normalized, cleanup_used, cleanup_reason = normalize_json_response(raw_response)
    if not normalized:
        return ParseResult(
            brief=None,
            error_message="Response was empty after cleanup.",
            reason="empty_response",
            cleanup_used=cleanup_used,
            cleanup_reason=cleanup_reason,
            raw_preview=_debug_preview(raw_response),
        )

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        return ParseResult(
            brief=None,
            error_message=f"Invalid JSON: {exc}",
            reason="json_decode_error",
            cleanup_used=cleanup_used,
            cleanup_reason=cleanup_reason,
            raw_preview=_debug_preview(raw_response),
        )

    if not isinstance(payload, dict):
        return ParseResult(
            brief=None,
            error_message="Expected a top-level JSON object.",
            reason="unexpected_top_level_shape",
            cleanup_used=cleanup_used,
            cleanup_reason=cleanup_reason,
            raw_preview=_debug_preview(raw_response),
        )

    try:
        return ParseResult(
            brief=LLMResearchBrief.model_validate(payload),
            error_message=None,
            reason=None,
            cleanup_used=cleanup_used,
            cleanup_reason=cleanup_reason,
        )
    except ValidationError as exc:
        reason, field = _validation_reason(exc)
        return ParseResult(
            brief=None,
            error_message=str(exc),
            reason=reason,
            field=field,
            cleanup_used=cleanup_used,
            cleanup_reason=cleanup_reason,
            raw_preview=_debug_preview(raw_response),
        )


def normalize_json_response(raw_response: str) -> tuple[str, bool, str | None]:
    text = raw_response.removeprefix("\ufeff").strip()
    cleanup_used = text != raw_response
    cleanup_reason = "bom_or_whitespace" if cleanup_used else None

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip(), True, "markdown_code_fence"

    if text.startswith("{"):
        try:
            _, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return text, cleanup_used
        trailing = text[end:].strip()
        if trailing:
            return text[:end].strip(), True, "surrounding_text"
        return text, cleanup_used, cleanup_reason

    object_start = text.find("{")
    if object_start >= 0:
        try:
            _, end = json.JSONDecoder().raw_decode(text[object_start:])
        except json.JSONDecodeError:
            return text, cleanup_used, cleanup_reason
        return text[object_start : object_start + end].strip(), True, "surrounding_text"

    return text, cleanup_used, cleanup_reason


def validate_source_references(
    brief: LLMResearchBrief,
    sources: list[ResearchSource],
) -> SanitizedResearchBrief:
    valid_ids = {source.id for source in sources}
    key_points: list[ResearchKeyPoint] = []
    for item in brief.key_points:
        source_ids = _valid_source_ids(item.source_ids, valid_ids)
        if len(source_ids) < len(item.source_ids):
            logger.info("research citation_filtered type=key_point reason=invalid_source_id")
        if not source_ids:
            logger.info("research citation_filtered type=key_point reason=no_valid_citations")
            continue
        key_points.append(ResearchKeyPoint(text=item.text, source_ids=source_ids))

    connections: list[ResearchConnection] = []
    for item in brief.connections:
        source_ids = _valid_source_ids(item.source_ids, valid_ids)
        if len(source_ids) < len(item.source_ids):
            logger.info("research citation_filtered type=connection reason=invalid_source_id")
        if not source_ids:
            logger.info("research citation_filtered type=connection reason=no_valid_citations")
            continue
        connections.append(
            ResearchConnection(
                concept=item.concept,
                explanation=item.explanation,
                source_ids=source_ids,
            )
        )
    return SanitizedResearchBrief(
        key_points=key_points,
        connections=connections,
        open_questions=[question for question in brief.open_questions if question.strip()],
        missing_information=[item for item in brief.missing_information if item.strip()],
    )


def _valid_source_ids(source_ids: list[str], valid_ids: set[str]) -> list[str]:
    seen: set[str] = set()
    filtered: list[str] = []
    for source_id in source_ids:
        if source_id in valid_ids and source_id not in seen:
            filtered.append(source_id)
            seen.add(source_id)
    return filtered


def _brief_messages(
    question: str,
    sources: list[ResearchSource],
    source_contents: dict[str, str],
    intent: QueryIntent,
    task_items: dict[str, list[TaskItem]],
) -> list[dict[str, str]]:
    allowed_source_ids = [source.id for source in sources]
    completion_rule = ""
    if intent in {QueryIntent.COMPLETION, QueryIntent.STATUS}:
        completion_rule = (
            " Only describe work as remaining when the source explicitly marks it incomplete "
            "or clearly states that it has not started or is unfinished. Do not treat completed "
            "tasks as remaining. Treat items with unknown completion status as uncertain, not "
            "incomplete. Do not treat concept lists, study topics, note names, or reading lists "
            "as unfinished work. Prefer milestones, deliverables, completion criteria, activities, "
            "and explicit status evidence."
        )
    return [
        {
            "role": "system",
            "content": (
                f"{NO_THINK_PREFIX}"
                "Return one JSON object only. Do not use Markdown fences. Do not write prose "
                "before or after JSON. Use only the retrieved context. Do not use outside "
                "knowledge. Use only these source IDs: "
                f"{', '.join(allowed_source_ids)}. Fewer items are acceptable. Empty arrays are "
                "valid. Put source IDs only in source_ids arrays, never in natural-language text. "
                "Do not include unsupported fields."
                f"{completion_rule}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Retrieved context:\n"
                f"{_format_source_context(sources, source_contents, task_items)}\n\n"
                "Required JSON shape:\n"
                f"{_schema_example(allowed_source_ids)}"
            ),
        },
    ]


def _repair_messages(
    sources: list[ResearchSource],
    raw_response: str,
    error: str | None,
) -> list[dict[str, str]]:
    allowed_source_ids = [source.id for source in sources]
    return [
        {
            "role": "system",
            "content": (
                f"{NO_THINK_PREFIX}"
                "Repair formatting only. Return one JSON object only. Do not use Markdown "
                "fences. Do not write prose before or after JSON. Use only these source IDs: "
                f"{', '.join(allowed_source_ids)}. Do not include unsupported fields."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Validation error: {error or 'unknown error'}\n\n"
                "Required JSON shape:\n"
                f"{_schema_example(allowed_source_ids)}\n\n"
                f"Invalid response:\n{raw_response[:2000]}"
            ),
        },
    ]


def _format_source_context(
    sources: list[ResearchSource],
    source_contents: dict[str, str],
    task_items: dict[str, list[TaskItem]] | None = None,
) -> str:
    parts: list[str] = []
    for source in sources:
        source_payload: dict[str, Any] = {
            "id": source.id,
            "file": source.file,
            "path": source.path,
            "heading": source.heading,
            "score": source.score,
            "excerpt": source.excerpt,
            "content": _content_with_task_state(
                source_contents.get(source.id, source.excerpt),
                task_items.get(source.id, []) if task_items else [],
            ),
        }
        parts.append(json.dumps(source_payload, ensure_ascii=True))
    return "\n".join(parts)


def _extract_json(raw_response: str) -> str:
    return normalize_json_response(raw_response)[0]


def _tasks_by_source(source_contents: dict[str, str]) -> dict[str, list[TaskItem]]:
    return {
        source_id: parse_task_items(content)
        for source_id, content in source_contents.items()
    }


def _content_with_task_state(content: str, tasks: list[TaskItem]) -> str:
    if not tasks:
        return content
    labeled_tasks = "\n".join(f"[{task.state}] {task.text}" for task in tasks)
    return f"{content}\n\nParsed task states:\n{labeled_tasks}"


def _apply_completion_state_awareness(
    brief: SanitizedResearchBrief,
    sources: list[ResearchSource],
    source_contents: dict[str, str],
    task_items: dict[str, list[TaskItem]],
) -> SanitizedResearchBrief:
    source_ids = {source.id for source in sources}
    complete_tasks = {
        _normalize_task_text(task.text)
        for tasks in task_items.values()
        for task in tasks
        if task.state == "complete"
    }
    key_points = [
        item
        for item in brief.key_points
        if not _completion_claim_relies_on_completed_task(item.text, complete_tasks)
    ]
    deterministic_key_points = _completion_key_points(sources, task_items, source_contents)
    key_points = deterministic_key_points + [
        item
        for item in key_points
        if not _duplicates_deterministic_key_point(item, deterministic_key_points)
    ]
    deduped_key_points: list[ResearchKeyPoint] = []
    seen_texts: set[str] = set()
    for item in key_points:
        if item.text not in seen_texts and set(item.source_ids) <= source_ids:
            deduped_key_points.append(item)
            seen_texts.add(item.text)
    return SanitizedResearchBrief(
        key_points=deduped_key_points,
        connections=brief.connections,
        open_questions=brief.open_questions,
        missing_information=brief.missing_information,
    )


def _completion_key_points(
    sources: list[ResearchSource],
    task_items: dict[str, list[TaskItem]],
    source_contents: dict[str, str],
) -> list[ResearchKeyPoint]:
    points: list[ResearchKeyPoint] = []
    for source in sources:
        heading_kind = classify_heading(source.heading or "")
        tasks = task_items.get(source.id, [])
        incomplete = [task.text for task in tasks if task.state == "incomplete"]
        unknown = [task.text for task in tasks if task.state == "unknown"]
        content = source_contents.get(source.id, "")
        if heading_kind == HeadingKind.LOW_VALUE_COMPLETION:
            continue
        if incomplete:
            points.append(
                ResearchKeyPoint(
                    text="Remaining: " + "; ".join(incomplete),
                    source_ids=[source.id],
                )
            )
            if unknown:
                points.append(
                    ResearchKeyPoint(
                        text="Completion status unclear: " + "; ".join(unknown),
                        source_ids=[source.id],
                    )
                )
            continue
        if heading_kind == HeadingKind.STATUS:
            status_text = _status_summary(content)
            if status_text:
                points.append(
                    ResearchKeyPoint(text=f"Current status: {status_text}", source_ids=[source.id])
                )
            continue
        if heading_kind == HeadingKind.COMPLETION_CRITERIA and unknown:
            unknown = _exclude_completed_roadmap_items(unknown)
            if not unknown:
                continue
            points.append(
                ResearchKeyPoint(
                    text="Upcoming milestones: " + "; ".join(unknown),
                    source_ids=[source.id],
                )
            )
            continue
        if heading_kind == HeadingKind.DELIVERABLES and unknown:
            unknown = _exclude_completed_roadmap_items(unknown)
            if not unknown:
                continue
            points.append(
                ResearchKeyPoint(
                    text="Future deliverables: " + "; ".join(unknown),
                    source_ids=[source.id],
                )
            )
            continue
        if heading_kind == HeadingKind.ACTIVITIES and unknown:
            unknown = _exclude_completed_roadmap_items(unknown)
            if not unknown:
                continue
            points.append(
                ResearchKeyPoint(
                    text="Planned work: " + "; ".join(unknown),
                    source_ids=[source.id],
                )
            )
            continue
        if heading_kind in {HeadingKind.ACTIVITIES} and unknown:
            points.append(
                ResearchKeyPoint(
                    text="Completion status unclear: " + "; ".join(unknown),
                    source_ids=[source.id],
                )
            )
        if not incomplete and has_explicit_incomplete_evidence(content):
            points.append(
                ResearchKeyPoint(
                    text="The source explicitly describes unfinished or in-progress work.",
                    source_ids=[source.id],
                )
            )
    return points


def _completion_claim_relies_on_completed_task(text: str, complete_tasks: set[str]) -> bool:
    normalized_text = _normalize_task_text(text)
    return any(task and task in normalized_text for task in complete_tasks)


def _normalize_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_]+", "", text.lower())).strip()


def _exclude_completed_roadmap_items(items: list[str]) -> list[str]:
    return [item for item in items if not _has_conservative_completion_word(item)]


def _has_conservative_completion_word(text: str) -> bool:
    return bool(
        re.search(
            r"\b(completed|complete|finished|done|implemented|created)\b",
            text,
            re.IGNORECASE,
        )
    )


def _duplicates_deterministic_key_point(
    candidate: ResearchKeyPoint,
    deterministic_points: list[ResearchKeyPoint],
) -> bool:
    candidate_tokens = _meaningful_tokens(candidate.text)
    if not candidate_tokens:
        return False
    for point in deterministic_points:
        if not set(candidate.source_ids) & set(point.source_ids):
            continue
        point_tokens = _meaningful_tokens(point.text)
        if not point_tokens:
            continue
        overlap = len(candidate_tokens & point_tokens) / min(len(candidate_tokens), len(point_tokens))
        if overlap >= 0.62:
            return True
    return False


def _meaningful_tokens(text: str) -> set[str]:
    normalized = re.sub(
        r"^(current status|upcoming milestones|future deliverables|planned work|remaining|completion status unclear):\s*",
        "",
        text.lower(),
    )
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {token for token in tokens if len(token) > 2 and token not in stopwords}


def _status_summary(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {"---"}:
            continue
        lines.append(stripped.lstrip("-*+ ").strip())
    return "; ".join(lines)


def _schema_example(allowed_source_ids: list[str]) -> str:
    source_id = allowed_source_ids[0] if allowed_source_ids else "source_1"
    return json.dumps(
        {
            "key_points": [
                {
                    "text": "Evidence-supported point.",
                    "source_ids": [source_id],
                }
            ],
            "connections": [
                {
                    "concept": "Concept name",
                    "explanation": "Evidence-supported relationship.",
                    "source_ids": [source_id],
                }
            ],
            "open_questions": ["Question the user may investigate next."],
            "missing_information": ["Information absent from the retrieved context."],
        },
        ensure_ascii=True,
    )


def _validation_reason(exc: ValidationError) -> tuple[str, str | None]:
    errors = exc.errors()
    if not errors:
        return "schema_validation_error", None

    first = errors[0]
    error_type = str(first.get("type", ""))
    location = first.get("loc", ())
    field = str(location[0]) if location else None
    if error_type == "missing":
        return "missing_field", field
    if error_type in {"list_type", "string_type", "model_type"}:
        return "invalid_field_type", field
    return "schema_validation_error", field


def _log_validation_failure(stage: str, result: ParseResult) -> None:
    if result.field:
        logger.info(
            "research %s_validation_failed reason=%s field=%s",
            stage,
            result.reason,
            result.field,
        )
    else:
        logger.info("research %s_validation_failed reason=%s", stage, result.reason)
    if result.raw_preview:
        logger.debug("research %s_raw_preview=%s", stage, result.raw_preview)


def _debug_preview(raw_response: str, limit: int = 500) -> str | None:
    if not logger.isEnabledFor(logging.DEBUG):
        return None
    return raw_response[:limit]
