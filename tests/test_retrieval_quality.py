from agent.retrieval.search import RetrievedChunk
from agent.services.retrieval_quality import (
    HeadingKind,
    QueryIntent,
    assess_retrieval_quality,
    classify_heading,
    detect_query_intent,
    rank_for_research,
    select_diverse_completion_chunks,
)


def test_completion_intent_boosts_administrative_headings() -> None:
    chunks = [
        _chunk(1, "Overview", "The project explores orbital habitability.", 0.80),
        _chunk(2, "Tasks", "- Finish simulation\n- Write summary", 0.70),
    ]

    ranked = rank_for_research(chunks, QueryIntent.COMPLETION, min_meaningful_content_chars=10)

    assert [chunk.heading for chunk in ranked] == ["Tasks", "Overview"]


def test_completion_intent_prioritizes_roadmap_headings_over_learn_sections() -> None:
    chunks = [
        _chunk(1, "Learn", "- Stellar luminosity", 0.95),
        _chunk(2, "Deliverables", "- Graph of habitable-zone movement", 0.50),
        _chunk(3, "Completion Criteria", "- Can explain habitable zone migration", 0.45),
        _chunk(4, "Status", "No simulation development has started.", 0.40),
    ]

    ranked = rank_for_research(chunks, QueryIntent.COMPLETION, min_meaningful_content_chars=10)

    assert [chunk.heading for chunk in ranked][:3] == [
        "Status",
        "Completion Criteria",
        "Deliverables",
    ]
    assert ranked[-1].heading == "Learn"


def test_completion_selection_is_category_diverse() -> None:
    chunks = [
        _chunk(1, "Completion Criteria", "- Criterion A", 3.0),
        _chunk(2, "Completion Criteria", "- Criterion B", 2.9),
        _chunk(3, "Completion Criteria", "- Criterion C", 2.8),
        _chunk(4, "Deliverables", "- Graph", 2.0),
        _chunk(5, "Activities", "- Plot results", 1.8),
        _chunk(6, "Status", "No simulation development has started.", 1.5),
    ]

    selected = select_diverse_completion_chunks(chunks, limit=5)

    headings = [chunk.heading for chunk in selected]
    assert headings[:4] == ["Status", "Completion Criteria", "Deliverables", "Activities"]
    assert headings.count("Completion Criteria") == 2


def test_completion_selection_does_not_fabricate_missing_categories() -> None:
    chunks = [
        _chunk(1, "Completion Criteria", "- Criterion A", 3.0),
        _chunk(2, "Completion Criteria", "- Criterion B", 2.9),
        _chunk(3, "Goal", "Build the model.", 2.0),
    ]

    selected = select_diverse_completion_chunks(chunks, limit=5)

    assert [chunk.heading for chunk in selected] == ["Completion Criteria", "Completion Criteria", "Goal"]


def test_heading_classification_is_case_insensitive() -> None:
    assert classify_heading("Current Status") == HeadingKind.STATUS
    assert classify_heading("success criteria") == HeadingKind.COMPLETION_CRITERIA
    assert classify_heading("Artifacts") == HeadingKind.DELIVERABLES
    assert classify_heading("Next Steps") == HeadingKind.ACTIVITIES
    assert classify_heading("Learn") == HeadingKind.LOW_VALUE_COMPLETION


def test_research_intent_penalizes_administrative_headings() -> None:
    chunks = [
        _chunk(1, "Checklist", "- Finish simulation\n- Write summary", 0.80),
        _chunk(2, "Methodology", "The model studies orbital habitability with simulation.", 0.75),
    ]

    ranked = rank_for_research(chunks, QueryIntent.RESEARCH, min_meaningful_content_chars=10)

    assert [chunk.heading for chunk in ranked] == ["Methodology", "Checklist"]


def test_heading_only_chunks_are_removed_from_ranking_candidates() -> None:
    chunks = [
        _chunk(1, "The Long Orbit", "# The Long Orbit", 0.99),
        _chunk(2, "Goal", "The goal is to study planetary habitability.", 0.70),
    ]

    ranked = rank_for_research(chunks, QueryIntent.RESEARCH, min_meaningful_content_chars=20)

    assert [chunk.heading for chunk in ranked] == ["Goal"]


def test_retrieval_assessment_rejects_low_relevance() -> None:
    assessment = assess_retrieval_quality(
        "Explain the French Revolution",
        [_chunk(1, "Astronomy", "Astronomy content.", 0.10)],
        min_top_score=0.35,
        min_average_score=0.25,
    )

    assert assessment.supported is False
    assert assessment.reason == "insufficient_relevance"


def test_detect_query_intent_completion() -> None:
    assert detect_query_intent("What remains to be completed in The Long Orbit?") == QueryIntent.COMPLETION


def _chunk(chunk_id: int, heading: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        filename="note.md",
        path="note.md",
        heading=heading,
        content=content,
        score=score,
    )
