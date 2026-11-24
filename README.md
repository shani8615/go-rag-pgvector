# go-rag-pgvector

[![CI](https://github.com/shani8615/go-rag-pgvector/actions/workflows/ci.yml/badge.svg)](https://github.com/shani8615/go-rag-pgvector/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/Go-1.24-00ADD8?logo=go)
![License](https://img.shields.io/badge/license-MIT-green)

**Chat with your documents.** A small, production-minded Retrieval-Augmented Generation (RAG) service in Go:
upload Markdown or text, ask questions, and get answers **grounded in your documents with source citations**,
or a clear *"I don't know"* when the answer isn't there.

- Go 1.24, standard library HTTP, one dependency (`pgx`)
- PostgreSQL + **pgvector** with an HNSW index for cosine similarity search
- Works with any **OpenAI-compatible** API (OpenAI, Ollama, vLLM, gateways)
- Guardrails enforced **in code**, not just in the prompt
- Unit tests, a pgvector integration test, and CI

```mermaid
flowchart LR
    subgraph Ingest
        A[POST /v1/documents] --> B[Chunk<br/>paragraph-aware, overlapping]
        B --> C[Embed]
        C --> D[(Postgres + pgvector<br/>HNSW index)]
    end
    subgraph Ask
        Q[POST /v1/ask] --> E[Embed question]
        E --> F[Top-k cosine search]
        D --> F
        F --> G{Any chunk above<br/>MIN_SCORE?}
        G -- no --> N["I don't know"<br/>no LLM call]
        G -- yes --> H[LLM answers from<br/>numbered passages only]
        H --> I{Cites a real<br/>passage?}
        I -- no --> N
        I -- yes --> R[Answer + sources]
    end
```

## Quick start

### Option 1: Docker Compose (Postgres + pgvector + API)

```bash
cp .env.example .env          # then set OPENAI_API_KEY in .env
docker compose up --build
```

### Option 2: Try it with no API key and no database

```bash
OFFLINE_DEMO=1 go run ./cmd/server
```

Offline mode uses keyword vectors and extractive answers instead of an LLM, so you can explore the API in seconds.
Use a real model for real results.

### Use it

```bash
# Index a document
curl -F file=@examples/acme-handbook.md localhost:8080/v1/documents
# {"chunks":1,"doc_id":"1","source":"acme-handbook.md"}

# Ask a question
curl localhost:8080/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"How long are backups kept?"}'
# {"answer":"Backups are kept for 35 days [1].","sources":[{"source":"acme-handbook.md","chunk_index":0,...}]}

# Ask something the documents don't cover
curl localhost:8080/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"Who is the CEO?"}'
# {"answer":"I don't know. I couldn't find this in the provided documents.","sources":[]}
```

## API

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/v1/documents` | multipart `file` (.md / .txt) **or** JSON `{"source","text"}` | Max 2 MiB. Stored atomically in one transaction. |
| `POST` | `/v1/ask` | JSON `{"question"}` | Max 1,000 characters. Returns `answer` and cited `sources`. |
| `GET` | `/healthz` | | Liveness check. |

## Design decisions

**Refuse in code, not just in the prompt.** Prompts alone don't reliably stop a model from making things up. Here:

1. If no chunk scores above `MIN_SCORE`, the service returns *"I don't know"* **without calling the LLM** (no cost, no hallucination).
2. The model sees numbered passages and must cite them, e.g. `[1]`.
3. Answers with no valid citation are rejected, and citations that point to passages that don't exist are ignored.

**Small interfaces, swappable parts.** The core (`internal/rag`) depends only on three interfaces: `Embedder`, `ChatModel`, `Store`.
Postgres, the in-memory store, OpenAI and the offline models are interchangeable, and that's how the tests run without network access.

**Sensible chunking.** Text is split on paragraphs, then sentences, with overlap between chunks so answers that span a boundary are still found.
It never splits a multi-byte UTF-8 character, which matters for Urdu, Arabic and other non-Latin text.

**Operational basics.** Request size limits, per-request timeouts, graceful shutdown, structured JSON logs (`log/slog`),
no secrets in the repo, and a distroless non-root container.

## Configuration

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | *(required unless offline)* | API key for the model provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `EMBED_MODEL` / `EMBED_DIM` | `text-embedding-3-small` / `1536` | `EMBED_DIM` must match the model |
| `CHAT_MODEL` | `gpt-4o-mini` | |
| `DATABASE_URL` | *(unset = in-memory)* | Postgres with the `vector` extension |
| `TOP_K` / `MIN_SCORE` | `5` / `0.30` | Retrieval tuning |
| `OFFLINE_DEMO` | | `1` = no API key needed |
| `PORT` | `8080` | |

## Project layout

```
cmd/server/          entry point: config, wiring, graceful shutdown
internal/rag/        chunking, retrieval, prompt, citation guardrails (no I/O)
internal/openai/     minimal OpenAI-compatible client (stdlib only)
internal/pgstore/    Postgres + pgvector store (pgx, HNSW cosine index)
internal/httpapi/    JSON HTTP API with input limits
examples/            sample document
```

## Tests

```bash
go test ./...                                   # unit tests, no network needed
docker compose up -d db
TEST_DATABASE_URL=postgres://rag:rag@localhost:5432/rag?sslmode=disable go test ./internal/pgstore/
```

## Roadmap

- PDF and DOCX ingestion
- Hybrid search (pgvector + Postgres full-text) with re-ranking
- Streaming answers (SSE)
- Per-tenant collections with row-level isolation
- An evaluation set with answer-quality metrics in CI

## License

MIT © Zeeshan Sarwar Awan
