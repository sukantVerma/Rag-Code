.PHONY: setup install run-backend run-frontend test docker-build docker-up docker-down

# ── Local setup ──────────────────────────────────────────────────────────────

setup:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example — add your ANTHROPIC_API_KEY"; fi
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt -r frontend/requirements.txt
	mkdir -p logs

install:
	.venv/bin/pip install -r requirements.txt -r frontend/requirements.txt

# ── Running locally ───────────────────────────────────────────────────────────

run-backend:
	mkdir -p logs
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee logs/backend.log

run-frontend:
	mkdir -p logs
	BACKEND_URL=http://localhost:8000 .venv/bin/streamlit run frontend/streamlit_app.py --server.port 8501 2>&1 | tee logs/frontend.log

logs-backend:
	tail -f logs/backend.log

logs-frontend:
	tail -f logs/frontend.log

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	.venv/bin/pytest tests/ -v

# ── Docker / Podman ───────────────────────────────────────────────────────────

docker-build:
	docker build -f Containerfile -t codedoc-rag:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
