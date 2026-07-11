from __future__ import annotations

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
    SearchRequest,
    SearchResponse,
    SearchResult,
    Source,
)
from agent.retrieval.research import ResearchService
from agent.retrieval.search import RetrievedChunk, SemanticSearch
from agent.services.ollama import OllamaClient, OllamaError


router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
    try:
        with connect(settings.database_path) as connection:
            search = SemanticSearch(connection, ollama)
            answer, chunks = await ResearchService(search, ollama).answer(payload.question, payload.top_k)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ResearchResponse(
        answer=answer,
        sources=[_to_source(chunk) for chunk in chunks],
    )


def _to_source(chunk: RetrievedChunk) -> Source:
    return Source(
        filename=chunk.filename,
        path=chunk.path,
        heading=chunk.heading,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
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
