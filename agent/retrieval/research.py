from __future__ import annotations

from agent.retrieval.search import RetrievedChunk, SemanticSearch
from agent.services.ollama import OllamaClient


class ResearchService:
    def __init__(self, search: SemanticSearch, ollama: OllamaClient) -> None:
        self.search = search
        self.ollama = ollama

    async def answer(self, question: str, top_k: int) -> tuple[str, list[RetrievedChunk]]:
        chunks = await self.search.search(question, top_k=top_k)
        if not chunks:
            return (
                "I could not find relevant indexed notes to answer that question.",
                [],
            )

        context = _format_context(chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Atlas, a local-first research assistant. Answer only from the "
                    "retrieved context. If the context does not contain the answer, say that "
                    "the indexed notes do not provide enough information. Do not invent facts "
                    "or citations."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ]
        answer = await self.ollama.chat(messages)
        return answer, chunks


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"File: {chunk.filename}",
                    f"Heading: {chunk.heading}",
                    chunk.content,
                ]
            )
        )
    return "\n\n".join(parts)
