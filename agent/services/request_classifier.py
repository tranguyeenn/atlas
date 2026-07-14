from __future__ import annotations

import re
from enum import StrEnum

from agent.services.ollama import OllamaClient


class RequestCategory(StrEnum):
    RESEARCH_ASSISTANCE = "research_assistance"
    GENERATIVE_WRITING = "generative_writing"
    UNSUPPORTED = "unsupported"


GENERATIVE_WRITING_MESSAGE = (
    "This request is primarily generative writing rather than research assistance. "
    "Atlas can help you find evidence, organize sources, identify connections, and "
    "develop questions. Use ChatGPT for general-purpose drafting."
)

UNSUPPORTED_MESSAGE = (
    "Atlas can only assist with research grounded in indexed notes. I could not find "
    "relevant indexed evidence for this request."
)
NO_THINK_PREFIX = "/no_think\n"


_GENERATIVE_PATTERNS = [
    r"\bwrite\s+(my|the|a|an)?\s*(complete|full|entire|final|submission-ready)?\s*"
    r"(research\s+paper|paper|essay|report|literature\s+review|conclusion|section)\b",
    r"\bproduce\s+(a|an|my|the)?\s*(complete|full|final|submission-ready)\s*"
    r"(literature\s+review|report|essay|paper)\b",
    r"\bdraft\s+(my|the|a|an)?\s*(complete|full|final)\s*"
    r"(paper|essay|report|conclusion|section)\b",
]

_RESEARCH_ASSISTANCE_PATTERNS = [
    r"\bwhat\s+do\s+my\s+notes\s+say\b",
    r"\bwhich\s+notes?\s+connect\b",
    r"\bwhat\s+evidence\b",
    r"\bwhat\s+remains\b",
    r"\bwhere\s+are\s+the\s+gaps\b",
    r"\bidentify\s+(connections|gaps|open\s+questions)\b",
    r"\bsurface\s+(evidence|sources|connections)\b",
    r"\bfind\s+(evidence|sources|notes|connections)\b",
    r"\b(completion|deliverables?|progress|status|unfinished|left\s+to\s+do)\b",
    r"\bquestions?\s+should\s+i\s+investigate\b",
    r"\bhow\s+does\b",
    r"\bhow\s+do\b",
    r"\bexplain\b",
    r"\bwhat\s+is\b",
    r"\bwhat\s+does\b",
    r"\bwhat\s+are\b",
]

_UNSUPPORTED_PATTERNS = [
    r"\b(current|latest|today|yesterday|this\s+week|this\s+month|news|web|internet)\b",
    r"\bsearch\s+the\s+web\b",
    r"\blook\s+up\b",
]


class RequestClassifier:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    async def classify(self, question: str) -> RequestCategory:
        rule_category = classify_with_rules(question)
        if rule_category is not None:
            return rule_category

        messages = [
            {
                "role": "system",
                "content": (
                    f"{NO_THINK_PREFIX}"
                    "Classify the user request as exactly one of: "
                    "research_assistance, generative_writing, unsupported. "
                    "research_assistance means finding evidence, connections, gaps, "
                    "or questions from indexed notes. generative_writing means asking "
                    "for finished essays, reports, papers, or submission-ready prose. "
                    "unsupported means requests requiring current web research or not "
                    "grounded in indexed notes. Return only the category string."
                ),
            },
            {"role": "user", "content": question},
        ]
        raw_category = (await self.ollama.chat(messages)).strip().lower()
        for category in RequestCategory:
            if raw_category == category.value:
                return category
        return RequestCategory.RESEARCH_ASSISTANCE


def classify_with_rules(question: str) -> RequestCategory | None:
    normalized = " ".join(question.lower().split())
    if _matches_any(normalized, _GENERATIVE_PATTERNS):
        return RequestCategory.GENERATIVE_WRITING
    if _matches_any(normalized, _UNSUPPORTED_PATTERNS):
        return RequestCategory.UNSUPPORTED
    if _matches_any(normalized, _RESEARCH_ASSISTANCE_PATTERNS):
        return RequestCategory.RESEARCH_ASSISTANCE
    return None


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
