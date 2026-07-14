# Atlas

Atlas is a local-first personal operating system for Trang Nguyen's knowledge,
projects, research, learning, planning, and decision support.

The long-term product direction is entity-first rather than file-first: Atlas
should reason about projects, tasks, deliverables, decisions, goals, concepts,
repositories, journal entries, events, people, and files as connected objects.
See [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) for the full product
specification.

Version 0.2 is the first implementation slice. It indexes an Obsidian Markdown
vault, stores note chunks and embeddings in SQLite, and returns structured
research briefs grounded only in retrieved source chunks through Ollama.

## Product philosophy

Atlas exists to help Trang understand:

- what they know
- what they are doing
- what they should do next
- what has changed
- why decisions were made
- what matters most right now

Research assistance is currently the implemented surface area, but it is not the
product boundary. Files and chunks are storage details; the intended model is a
structured personal knowledge graph with traceable source evidence.

## Research-assistance philosophy

Atlas assists the user's research process without replacing the user's reasoning.
It can retrieve relevant notes, surface evidence, identify connections, suggest
open questions, expose gaps in available information, and cite every source used.

Atlas does not write complete essays or reports, produce submission-ready
academic work, claim to have completed research independently, make unsupported
conclusions, or replace the user's interpretation of evidence. When a request is
primarily asking for finished writing, Atlas redirects the user to ChatGPT for
general-purpose drafting.

## Requirements

- Python 3.11+
- Ollama running locally
- Ollama models:
  - `qwen3:4b`
  - `embeddinggemma:latest`

```bash
ollama pull qwen3:4b
ollama pull embeddinggemma:latest
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Find your Obsidian vault path

Atlas needs the folder path for your Obsidian vault, not the path to a single
note.

In Obsidian, open your vault and reveal or open the vault folder in your file
manager. On macOS, you can then right-click the vault folder in Finder, hold
Option, and choose **Copy ... as Pathname**.

### Configure Atlas

The easiest setup path is the interactive helper:

```bash
python3 -m agent.cli configure-obsidian
```

Paste the full vault folder path when prompted. The helper validates that the
folder exists and saves `OBSIDIAN_VAULT_PATH` in `.env`.

You can also edit `.env` manually:

```bash
OBSIDIAN_VAULT_PATH="/full/path/to/your/obsidian/vault"
OLLAMA_CHAT_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
```

Do not hardcode someone else's example path. Use the path to your own vault
folder.

Before indexing, confirm Atlas can read the configured vault path:

```bash
python3 -m agent.cli test-obsidian
```

## Run

```bash
uvicorn main:app --reload
```

At startup Atlas logs the non-sensitive model configuration that is passed to
the Ollama client:

```text
Atlas chat model: qwen3:4b
Atlas embedding model: embeddinggemma:latest
```

## API

Check runtime status and active models:

```bash
curl http://localhost:8000/health
```

Example:

```json
{
  "status": "ok",
  "version": "0.2.0",
  "models": {
    "chat": "qwen3:4b",
    "embedding": "embeddinggemma:latest"
  }
}
```

Index the configured vault:

```bash
curl -X POST http://localhost:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"force": false}'
```

For the first index, make sure Ollama is running, start Atlas with the command
above, then run the `/index` request. The first run may take a while because
Atlas reads your Markdown notes and creates embeddings locally.

Search indexed notes:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "What did I write about retrieval?", "top_k": 5}'
```

Ask a sourced research question:

```bash
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question": "What did I decide about memory?", "top_k": 5}'
```

`/research` classifies the request before generating a brief. Research-assistance
requests return structured evidence, connections, open questions, missing
information, and cited sources:

```json
{
  "status": "ok",
  "category": "research_assistance",
  "question": "How does The Long Orbit connect computer science, mathematics, and astronomy?",
  "key_points": [
    {
      "text": "The project uses computational modeling to simulate long-term planetary habitability.",
      "source_ids": ["source_1"]
    }
  ],
  "connections": [
    {
      "concept": "Orbital mechanics",
      "explanation": "Orbital mechanics provides mathematical models used in the simulations.",
      "source_ids": ["source_1"]
    }
  ],
  "open_questions": [
    "Which variables have the largest effect on long-term habitability?"
  ],
  "missing_information": [
    "The indexed notes do not specify which numerical integration method will be used."
  ],
  "sources": [
    {
      "id": "source_1",
      "file": "The Long Orbit.md",
      "path": "Projects/The Long Orbit.md",
      "heading": "Methodology",
      "score": 0.91,
      "excerpt": "The project uses computational modeling to simulate..."
    }
  ]
}
```

Every key point and connection must cite source IDs that exist in the returned
`sources` array. Atlas validates model output and attempts one JSON repair before
returning a controlled error response.

Requests for finished writing return a redirect response:

```json
{
  "status": "redirect",
  "category": "generative_writing",
  "message": "This request is primarily generative writing rather than research assistance. Atlas can help you find evidence, organize sources, identify connections, and develop questions. Use ChatGPT for general-purpose drafting."
}
```

Unsupported requests, including requests with no relevant indexed evidence,
return:

```json
{
  "status": "unsupported",
  "category": "unsupported",
  "question": "Explain the French Revolution.",
  "message": "Atlas could not find relevant evidence in the indexed notes."
}
```

Before generating a research brief, Atlas applies retrieval-quality checks using
the top similarity score, average score, lightweight evidence overlap, and the
amount of meaningful non-heading content. It also detects research, status, and
completion intent. Administrative headings such as `Checklist`, `Status`,
`Deliverables`, and `Completion Criteria` are boosted for completion/status
questions and slightly penalized for ordinary research questions.

For completion/status questions, Atlas parses Markdown task markers before
formatting context for the model:

```md
- [ ] incomplete task
- [x] complete task
- [X] complete task
- plain bullet with unknown completion status
```

Completed tasks are not treated as remaining work. Plain bullets are surfaced as
unclear when relevant, not as unfinished.

The retrieval thresholds can be tuned in `.env`:

```bash
MIN_TOP_SCORE=0.35
MIN_AVERAGE_SCORE=0.25
MIN_MEANINGFUL_CONTENT_CHARS=40
```

`/search` remains a direct retrieval endpoint. It returns retrieved chunks and
metadata without creating a research brief.

Research requests also emit lightweight timing logs:

```text
research classification_seconds=0.001
research intent=research
research retrieval_seconds=0.120
research retrieval_supported=true
research retrieval_confidence=0.810
research retrieval_reason=supported
research generation_seconds=8.441
research repair_attempted=false
research total_seconds=8.580
```

## Tests

```bash
pytest
```

The test suite mocks Ollama calls and does not require a running Ollama server.
