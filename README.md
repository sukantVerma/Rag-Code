# CodeDoc RAG System

A production-ready **Multi-Source Retrieval-Augmented Generation** system that ingests GitHub repositories and PDF documents, stores their embeddings in vector databases, and answers natural language questions using Anthropic Claude.

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
│   OpenAI text-embedding-3-small  │
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
       │  (claude-sonnet-4-20250514) │
       └───────────────┘
```

## Quick Start

### 1. Clone & configure

```bash
git clone <this-repo>
cd codedoc-rag
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Using Podman (container)

```bash
podman-compose up --build
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
├── scripts/                   # CLI ingestion script (for CI/CD)
├── data/                      # Persisted vector stores, PDFs, cloned repos
├── tests/                     # Unit and integration tests
├── Containerfile              # Podman container definition
├── docker-compose.yml         # Podman-compatible compose file
└── requirements.txt
```

## CI/CD

On every push to `main`, the GitHub Actions workflow:

1. Checks out the repo (with `fetch-depth: 2`)
2. Computes changed files via `git diff HEAD~1 HEAD`
3. Runs incremental ingestion for only the changed files

Required GitHub Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GITHUB_PAT`.

## Running Tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `GITHUB_PAT` | GitHub personal access token (optional, for private repos) |
| `CHROMA_DB_PATH` | Path to persist ChromaDB (default: `./data/chroma_db`) |
| `FAISS_INDEX_PATH` | Path to persist FAISS index (default: `./data/faiss_index`) |
| `PDF_DATA_PATH` | Directory for uploaded PDFs (default: `./data/pdfs`) |
| `REPO_CLONE_PATH` | Directory for cloned repos (default: `./data/repos`) |

## License

MIT
