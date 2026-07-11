from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkdownChunk:
    heading: str
    content: str


def chunk_markdown_by_headings(markdown: str) -> list[MarkdownChunk]:
    chunks: list[MarkdownChunk] = []
    current_heading = "Untitled"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if _is_heading(line):
            _append_chunk(chunks, current_heading, current_lines)
            current_heading = line.lstrip("#").strip() or "Untitled"
            current_lines = [line]
        else:
            current_lines.append(line)

    _append_chunk(chunks, current_heading, current_lines)
    return chunks


def _is_heading(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#") and len(stripped) > 1 and stripped[1] in {"#", " "}


def _append_chunk(chunks: list[MarkdownChunk], heading: str, lines: list[str]) -> None:
    content = "\n".join(lines).strip()
    if content:
        chunks.append(MarkdownChunk(heading=heading, content=content))
