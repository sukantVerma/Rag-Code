# CodeDoc RAG System

A **Multi-Source Retrieval-Augmented Generation** system that ingests GitHub repositories and PDF documents, stores their embeddings in vector databases, and answers natural language questions using Anthropic Claude.

## Architecture

```
┌─────────────┐    ┌─────────────┐
│  GitHub Repo │    │   PDF Docs  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌─────────────┐
│ Code Chunker │   │ PDF Chunker │
│ (LangChain)  │   │  (PyMuPDF)  │
└──────┬───────┘   └──────┬──────┘
       │                  │
       ▼                  ▼
┌──────────────────────────────────┐
│  fastembed (BAAI/bge-small-en)   │
│  local embeddings, no API key    │
└──────────────┬───────────────────┘
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌────────────┐
│  ChromaDB  │   │   FAISS    │
│  (code)    │   │  (PDFs)    │
└──────┬─────┘   └──────┬─────┘
       └───────┬─────────┘
               ▼
       ┌───────────────┐
       │  Multi-Source  │
       │  Retriever     │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │  Claude LLM   │
       │  (Anthropic)  │
       └───────────────┘
```

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com)
- Git

### 1. Set up environment

```bash
cd codedoc-rag
make setup
# Opens .env — add your ANTHROPIC_API_KEY
```

Or manually:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r frontend/requirements.txt
```

### 2. Run the backend

```bash
make run-backend
# FastAPI available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 3. Run the frontend (separate terminal)

```bash
make run-frontend
# Streamlit UI at http://localhost:8501
```

## Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

make docker-up
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

## API Endpoints

### Ingest a GitHub repository

```bash
curl -X POST http://localhost:8000/ingest/github \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/owner/repo"}'
```

### Ingest a PDF document

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the authentication module work?", "top_k": 10}'
```

### Health check

```bash
curl http://localhost:8000/health
```

## Running Tests

```bash
make test
# or: .venv/bin/pytest tests/ -v
```

## Project Structure

```
codedoc-rag/
├── .github/workflows/         # CI/CD pipeline
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # Environment-based settings
│   ├── ingestion/             # GitHub + PDF ingestion, chunking, embedding
│   ├── vectorstore/           # ChromaDB + FAISS stores
│   ├── retrieval/             # Multi-source retriever + reranker
│   ├── llm/                   # Claude API client
│   └── api/                   # FastAPI routes and schemas
├── frontend/
│   └── streamlit_app.py       # Streamlit chat UI
├── scripts/                   # CLI ingestion script (for CI/CD)
├── data/                      # Persisted vector stores, PDFs, cloned repos
├── tests/                     # Unit tests
├── Containerfile              # Container definition (Docker/Podman)
├── docker-compose.yml         # Multi-service orchestration
├── Makefile                   # Dev shortcuts
└── requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `GITHUB_PAT` | No | GitHub PAT for private repos |
| `ANTHROPIC_MODEL` | No | Claude model (default: `claude-sonnet-4-20250514`) |
| `CHROMA_DB_PATH` | No | ChromaDB path (default: `./data/chroma_db`) |
| `FAISS_INDEX_PATH` | No | FAISS index path (default: `./data/faiss_index`) |
| `PDF_DATA_PATH` | No | PDF upload directory (default: `./data/pdfs`) |
| `REPO_CLONE_PATH` | No | Repo clone directory (default: `./data/repos`) |
| `BACKEND_URL` | No | Backend URL for frontend (default: `http://localhost:8000`) |

## CI/CD

On every push to `main`, the GitHub Actions workflow:

1. Checks out the repo (with `fetch-depth: 2`)
2. Computes changed files via `git diff HEAD~1 HEAD`
3. Runs incremental ingestion for only the changed files

Required GitHub Secrets: `ANTHROPIC_API_KEY`, `GITHUB_PAT`.

## License

MIT
