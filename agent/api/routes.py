from __future__ import annotations

import logging
from time import perf_counter
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from agent.config import ObsidianVaultPathError, Settings, validate_obsidian_vault_path
from agent.database import connect
from agent.indexing.obsidian import ObsidianIndexer
from agent.models.schemas import (
    IndexRequest,
    IndexResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchRedirectResponse,
    ResearchUnsupportedResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from agent.retrieval.search import RetrievedChunk, SemanticSearch
from agent.services.request_classifier import (
    GENERATIVE_WRITING_MESSAGE,
    UNSUPPORTED_MESSAGE,
    RequestCategory,
    RequestClassifier,
)
from agent.services.research_brief import ResearchBriefService
from agent.services.retrieval_quality import (
    QueryIntent,
    assess_retrieval_quality,
    detect_query_intent,
    rank_for_research,
    select_diverse_completion_chunks,
)
from agent.services.source_formatter import format_sources
from agent.services.ollama import OllamaClient, OllamaError
from agent.services.project_state import ProjectStateService, detect_project_state_intent


router = APIRouter()
logger = logging.getLogger("atlas")
RETRIEVAL_UNSUPPORTED_MESSAGE = "Atlas could not find relevant evidence in the indexed notes."


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "version": request.app.version,
        "models": {
            "chat": settings.chat_model,
            "embedding": settings.embedding_model,
        },
    }


@router.post("/index", response_model=IndexResponse)
async def index_vault(
    payload: IndexRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
) -> IndexResponse:
    vault_path = Path(payload.vault_path).expanduser() if payload.vault_path else settings.obsidian_vault_path

    try:
        vault_path = validate_obsidian_vault_path(vault_path)
        with connect(settings.database_path) as connection:
            stats = await ObsidianIndexer(connection, ollama).index_vault(vault_path, force=payload.force)
    except (ObsidianVaultPathError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return IndexResponse(
        indexed_files=stats.indexed_files,
        skipped_files=stats.skipped_files,
        indexed_chunks=stats.indexed_chunks,
    )


@router.post("/search", response_model=SearchResponse)
async def search_notes(
    payload: SearchRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
) -> SearchResponse:
    try:
        with connect(settings.database_path) as connection:
            chunks = await SemanticSearch(connection, ollama).search(payload.query, payload.top_k)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SearchResponse(results=[_to_search_result(chunk) for chunk in chunks])


@router.post("/research", response_model=ResearchResponse)
async def research(
    payload: ResearchRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
) -> ResearchResponse:
    total_started_at = perf_counter()
    project_state_intent = detect_project_state_intent(payload.question)
    if project_state_intent is not None:
        with connect(settings.database_path) as connection:
            project_state_response = ProjectStateService(connection).answer(
                payload.question,
                project_state_intent,
            )
        logger.info("research project_state_intent=%s", project_state_intent.value)
        if project_state_response is not None:
            logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
            return project_state_response
        if project_state_intent.value in {"next_task", "project_summary", "phase", "decision", "blocked"}:
            logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
            return ResearchUnsupportedResponse(
                question=payload.question,
                message="Atlas could not find indexed project-state entities for this request.",
            )

    try:
        classification_started_at = perf_counter()
        category = await RequestClassifier(ollama).classify(payload.question)
        logger.info(
            "research classification_seconds=%.3f",
            perf_counter() - classification_started_at,
        )
        if category == RequestCategory.GENERATIVE_WRITING:
            logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
            return ResearchRedirectResponse(message=GENERATIVE_WRITING_MESSAGE)
        if category == RequestCategory.UNSUPPORTED:
            logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
            return ResearchUnsupportedResponse(question=payload.question, message=UNSUPPORTED_MESSAGE)

        intent = detect_query_intent(payload.question)
        logger.info("research intent=%s", intent.value)
        retrieval_started_at = perf_counter()
        with connect(settings.database_path) as connection:
            retrieved_chunks = await SemanticSearch(connection, ollama).search(
                payload.question,
                min(payload.top_k * 3, 20),
            )
        chunks = rank_for_research(
            retrieved_chunks,
            intent,
            settings.min_meaningful_content_chars,
        )
        if intent == QueryIntent.COMPLETION:
            chunks = select_diverse_completion_chunks(chunks, payload.top_k)
        else:
            chunks = chunks[: payload.top_k]
        assessment = assess_retrieval_quality(
            payload.question,
            chunks,
            settings.min_top_score,
            settings.min_average_score,
        )
        logger.info("research retrieval_seconds=%.3f", perf_counter() - retrieval_started_at)
        logger.info("research retrieval_supported=%s", str(assessment.supported).lower())
        logger.info("research retrieval_confidence=%.3f", assessment.confidence)
        logger.info("research retrieval_reason=%s", assessment.reason)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not assessment.supported:
        logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
        return ResearchUnsupportedResponse(
            question=payload.question,
            message=RETRIEVAL_UNSUPPORTED_MESSAGE,
        )

    sources = format_sources(chunks, settings.obsidian_vault_path)
    source_contents = {
        source.id: chunk.content
        for source, chunk in zip(sources, chunks, strict=True)
    }
    try:
        response = await ResearchBriefService(ollama).create_brief(
            payload.question,
            sources,
            source_contents,
            intent,
        )
        if _is_empty_absence_brief(response):
            logger.info("research empty_absence_brief_converted=true")
            logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
            return ResearchUnsupportedResponse(
                question=payload.question,
                message=RETRIEVAL_UNSUPPORTED_MESSAGE,
            )
        logger.info("research total_seconds=%.3f", perf_counter() - total_started_at)
        return response
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _is_empty_absence_brief(response: ResearchResponse) -> bool:
    if not hasattr(response, "key_points") or not hasattr(response, "connections"):
        return False
    if response.key_points or response.connections:
        return False
    absence_terms = ("absent", "not found", "no evidence", "do not mention", "does not mention")
    return any(
        any(term in item.lower() for term in absence_terms)
        for item in response.missing_information
    )


def _to_search_result(chunk: RetrievedChunk) -> SearchResult:
    return SearchResult(
        filename=chunk.filename,
        path=chunk.path,
        heading=chunk.heading,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        content=chunk.content,
    )
