from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    vault_path: str | None = None
    force: bool = False


class IndexResponse(BaseModel):
    indexed_files: int
    skipped_files: int
    indexed_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    filename: str
    path: str
    heading: str
    chunk_id: int
    score: float | None = None


class SearchResult(Source):
    content: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ResearchResponse(BaseModel):
    answer: str
    sources: list[Source]
