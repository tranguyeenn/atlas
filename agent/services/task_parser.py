from __future__ import annotations

import re

from agent.models.schemas import TaskItem


TASK_LINE_PATTERN = re.compile(
    r"^\s*[-*+]\s+(?:\[(?P<marker>[ xX])\]\s+)?(?P<text>.+?)\s*$"
)

INCOMPLETE_EVIDENCE_PATTERN = re.compile(
    r"\b(not started|has not started|in progress|unfinished|remaining|pending|not completed)\b",
    re.IGNORECASE,
)


def parse_task_items(markdown: str) -> list[TaskItem]:
    tasks: list[TaskItem] = []
    for line in markdown.splitlines():
        match = TASK_LINE_PATTERN.match(line)
        if not match:
            continue
        text = match.group("text").strip()
        marker = match.group("marker")
        if marker is None:
            state = "unknown"
        elif marker == " ":
            state = "incomplete"
        else:
            state = "complete"
        tasks.append(TaskItem(text=text, state=state))
    return tasks


def has_explicit_incomplete_evidence(text: str) -> bool:
    return bool(INCOMPLETE_EVIDENCE_PATTERN.search(text))
