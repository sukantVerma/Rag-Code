FROM python:3.11-slim

# Create a non-root user (Podman rootless best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install curl (health check polling in start.sh) and git (GitPython)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Install backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies
COPY frontend/requirements.txt ./frontend-requirements.txt
RUN pip install --no-cache-dir -r frontend-requirements.txt

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY start.sh .

RUN chmod +x start.sh

# Create data directories and set ownership
RUN mkdir -p data/chroma_db data/faiss_index data/pdfs data/repos

EXPOSE 8000
EXPOSE 8501

CMD ["./start.sh"]
