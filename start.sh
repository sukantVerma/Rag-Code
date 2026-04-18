#!/bin/bash
# Ensure data subdirectories exist (volume mount may be empty)
mkdir -p data/chroma_db data/faiss_index data/pdfs data/repos

# Start FastAPI backend in background on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready (poll /health)
echo "Waiting for backend to start..."
for i in {1..30}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend is ready!"
    break
  fi
  sleep 1
done

# Start Streamlit frontend on port 8501
streamlit run frontend/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false

# If streamlit exits, kill backend too
kill $BACKEND_PID
