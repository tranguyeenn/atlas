# Atlas

Atlas is a local-first AI research assistant for macOS. Version 0.1 indexes an
Obsidian Markdown vault, stores note chunks and embeddings in SQLite, and answers
questions using only retrieved source chunks through Ollama.

## Requirements

- Python 3.11+
- Ollama running locally
- Ollama models:
  - `qwen3:8b`
  - `embeddinggemma`

```bash
ollama pull qwen3:8b
ollama pull embeddinggemma
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

## API

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

Every `/research` response returns source filenames, paths, headings, chunk IDs,
and similarity scores.

## Tests

```bash
pytest
```
