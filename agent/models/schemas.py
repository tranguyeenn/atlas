from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class ResearchSource(BaseModel):
    id: str
    file: str
    path: str
    heading: str | None
    score: float
    excerpt: str


class TaskItem(BaseModel):
    text: str
    state: Literal["incomplete", "complete", "unknown"]


class ResearchKeyPoint(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("key points must cite at least one source")
        return value


class ResearchConnection(BaseModel):
    concept: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("connections must cite at least one source")
        return value


class LLMResearchKeyPoint(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class LLMResearchConnection(BaseModel):
    concept: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class LLMResearchBrief(BaseModel):
    key_points: list[LLMResearchKeyPoint] = Field(default_factory=list)
    connections: list[LLMResearchConnection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    status: Literal["ok"] = "ok"
    category: Literal["research_assistance"] = "research_assistance"
    question: str
    key_points: list[ResearchKeyPoint]
    connections: list[ResearchConnection]
    open_questions: list[str]
    missing_information: list[str]
    sources: list[ResearchSource]


class ResearchRedirectResponse(BaseModel):
    status: Literal["redirect"] = "redirect"
    category: Literal["generative_writing"] = "generative_writing"
    message: str


class ResearchUnsupportedResponse(BaseModel):
    status: Literal["unsupported"] = "unsupported"
    category: Literal["unsupported"] = "unsupported"
    question: str | None = None
    message: str


class RetrievalAssessment(BaseModel):
    supported: bool
    confidence: float
    reason: Literal["insufficient_relevance", "insufficient_evidence", "supported"]


class ResearchErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    category: Literal["research_assistance"] = "research_assistance"
    message: str


class ProjectStateEntity(BaseModel):
    id: int
    type: str
    name: str
    state: str | None = None
    project: str | None = None


class ProjectStateResponse(BaseModel):
    status: Literal["ok"] = "ok"
    category: Literal["project_state"] = "project_state"
    question: str
    answer: str
    recommended_action: str | None = None
    entities: list[ProjectStateEntity]
    missing_information: list[str]
    sources: list[ResearchSource]


ResearchResponse = (
    ResearchBrief
    | ResearchRedirectResponse
    | ResearchUnsupportedResponse
    | ResearchErrorResponse
    | ProjectStateResponse
)
