from agent.indexing.chunker import chunk_markdown_by_headings


def test_chunk_markdown_by_headings_keeps_preamble() -> None:
    markdown = "intro text\n\n# First\nbody\n\n## Second\nmore"

    chunks = chunk_markdown_by_headings(markdown)

    assert [chunk.heading for chunk in chunks] == ["Untitled", "First", "Second"]
    assert chunks[0].content == "intro text"
    assert chunks[1].content == "# First\nbody"
    assert chunks[2].content == "## Second\nmore"


def test_chunk_markdown_by_headings_ignores_empty_chunks() -> None:
    markdown = "\n\n# First\n\n# Second\ncontent"

    chunks = chunk_markdown_by_headings(markdown)

    assert [chunk.heading for chunk in chunks] == ["First", "Second"]
