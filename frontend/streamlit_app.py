"""
CodeDoc RAG — Streamlit Frontend

Single-file Streamlit app that provides a chat interface over the
CodeDoc RAG FastAPI backend.  Supports GitHub repo ingestion, PDF
upload, and natural-language Q&A with source attribution.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Language → badge colour mapping
LANG_COLORS: dict[str, str] = {
    "python": "#3572A5",
    "javascript": "#f1e05a",
    "typescript": "#3178c6",
    "java": "#b07219",
    "go": "#00ADD8",
    "rust": "#dea584",
}
DEFAULT_LANG_COLOR = "#808080"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CodeDoc RAG",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{role, content, sources, stats}]

if "sources_ingested" not in st.session_state:
    st.session_state.sources_ingested = {
        "github": None,  # {url, repo_name, chunks}
        "pdfs": [],      # [{file_name, chunks}]
    }

if "health_status" not in st.session_state:
    st.session_state.health_status = {"status": "unknown", "chroma": False, "faiss": False}

# ---------------------------------------------------------------------------
# Helper: call backend safely
# ---------------------------------------------------------------------------


def _api_get(path: str, **kwargs) -> requests.Response | None:
    """GET request to the backend; returns None on connection failure."""
    try:
        return requests.get(f"{API_URL}{path}", timeout=10, **kwargs)
    except requests.ConnectionError:
        return None


def _api_post(path: str, **kwargs) -> requests.Response | None:
    """POST request to the backend; returns None on connection failure."""
    try:
        return requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        return None


# ---------------------------------------------------------------------------
# Health check (runs on every rerun, cached briefly)
# ---------------------------------------------------------------------------


def _refresh_health() -> None:
    resp = _api_get("/health")
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        st.session_state.health_status = {
            "status": "ok",
            "chroma": bool(data.get("chroma")),
            "faiss": bool(data.get("faiss")),
        }
    else:
        st.session_state.health_status = {"status": "down", "chroma": False, "faiss": False}


_refresh_health()

# ---------------------------------------------------------------------------
# Sidebar — source management
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Title + health status ────────────────────────────────────────────
    st.title("🔍 CodeDoc RAG")

    health = st.session_state.health_status
    if health["status"] == "ok":
        st.markdown("🟢 **Backend connected**")
    else:
        st.markdown("🔴 **Backend unavailable**")

    chroma_icon = "✅" if health.get("chroma") else "❌"
    faiss_icon = "✅" if health.get("faiss") else "❌"
    st.caption(f"ChromaDB {chroma_icon}  &nbsp;|&nbsp;  FAISS {faiss_icon}")

    st.divider()

    # ── GitHub repository ingestion ──────────────────────────────────────
    st.subheader("🐙 GitHub Repository")

    repo_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
    pat_token = st.text_input("PAT Token (optional)", type="password")

    if st.button("Ingest Repo", type="primary", use_container_width=True):
        if not repo_url.strip():
            st.error("Please enter a repository URL.")
        else:
            with st.spinner("Cloning and embedding..."):
                resp = _api_post(
                    "/ingest/github",
                    json={
                        "repo_url": repo_url.strip(),
                        "pat_token": pat_token.strip() or None,
                    },
                )
            if resp is None:
                st.error(f"❌ Cannot connect to backend at {API_URL}. Is it running?")
            elif resp.status_code == 200:
                data = resp.json()
                repo_name = data.get("repo_name", repo_url.split("/")[-1])
                chunks = data.get("chunks_added", data.get("chunks_processed", 0))
                st.session_state.sources_ingested["github"] = {
                    "url": repo_url,
                    "repo_name": repo_name,
                    "chunks": chunks,
                }
                st.success(f"✅ **{repo_name}** — {chunks} chunks indexed")
                st.rerun()
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                st.error(f"❌ Ingestion failed: {detail}")

    # Show previously ingested repo
    gh = st.session_state.sources_ingested["github"]
    if gh:
        st.info(f"✅ **{gh['repo_name']}** — {gh['chunks']} chunks")

    st.divider()

    # ── PDF document ingestion ───────────────────────────────────────────
    st.subheader("📄 PDF Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if st.button("Ingest PDFs", use_container_width=True) and uploaded_files:
        progress = st.progress(0, text="Ingesting PDFs...")
        total = len(uploaded_files)

        for idx, uploaded in enumerate(uploaded_files):
            progress.progress(
                (idx) / total,
                text=f"Ingesting {uploaded.name} ({idx + 1}/{total})...",
            )
            resp = _api_post(
                "/ingest/pdf",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
            if resp is None:
                st.error(f"❌ Cannot connect to backend at {API_URL}.")
                break
            elif resp.status_code == 200:
                data = resp.json()
                chunks = data.get("chunks_added", data.get("chunks_processed", 0))
                st.session_state.sources_ingested["pdfs"].append(
                    {"file_name": uploaded.name, "chunks": chunks}
                )
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                st.error(f"❌ Failed to ingest {uploaded.name}: {detail}")

        progress.progress(1.0, text="Done!")
        st.rerun()

    # Show previously ingested PDFs
    for pdf in st.session_state.sources_ingested["pdfs"]:
        st.caption(f"✅ {pdf['file_name']} — {pdf['chunks']} chunks")

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        top_k = st.slider("Sources to retrieve", min_value=3, max_value=20, value=10)
        show_sources = st.toggle("Show sources", value=True)
        show_scores = st.toggle("Show similarity scores", value=True)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# ── Header ───────────────────────────────────────────────────────────────────

col_header, col_clear = st.columns([8, 2])

with col_header:
    st.header("Ask about your codebase and documents")

    # Build knowledge-base status line
    gh = st.session_state.sources_ingested["github"]
    pdfs = st.session_state.sources_ingested["pdfs"]
    parts: list[str] = []
    if gh:
        parts.append(gh["repo_name"])
    if pdfs:
        parts.append(f"{len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''}")
    if parts:
        st.caption(f"📚 Knowledge base: {' + '.join(parts)}")
    else:
        st.warning("⚠️ No sources ingested yet — use the sidebar to add a repo or PDFs.")

with col_clear:
    st.write("")  # spacer
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            sources = msg["sources"]
            stats = msg.get("stats", {})
            retrieval_count = stats.get("retrieval_count", len(sources))

            with st.expander(f"📎 {retrieval_count} sources retrieved — click to expand"):
                for src in sources:
                    source_type = src.get("source_type", src.get("source", "unknown"))
                    score = src.get("similarity_score", src.get("score"))

                    col_content, col_score = st.columns([3, 1])

                    with col_content:
                        if source_type == "github":
                            file_path = src.get("file_path", "unknown")
                            lang = src.get("language", "")
                            color = LANG_COLORS.get(lang, DEFAULT_LANG_COLOR)
                            st.markdown(
                                f"📄 **{file_path}** "
                                f'<span style="background-color:{color};color:#fff;'
                                f'padding:2px 8px;border-radius:4px;font-size:0.8em;">'
                                f"{lang}</span>",
                                unsafe_allow_html=True,
                            )
                            preview = src.get("content_preview", "")
                            if preview:
                                st.code(preview, language=lang or None)
                        else:
                            file_name = src.get("file_name", src.get("file_path", "unknown"))
                            page = src.get("page", src.get("page_number", "?"))
                            st.markdown(f"📕 **{file_name}** — page {page}")
                            preview = src.get("content_preview", "")
                            if preview:
                                st.markdown(f"> {preview}")

                    with col_score:
                        if show_scores and score is not None:
                            st.metric("Score", f"{score:.2f}")
                            if score >= 0.85:
                                bar_color = "#2ecc71"
                            elif score >= 0.70:
                                bar_color = "#f39c12"
                            else:
                                bar_color = "#e74c3c"
                            st.markdown(
                                f'<div style="background:#eee;border-radius:4px;height:8px;">'
                                f'<div style="width:{score * 100:.0f}%;background:{bar_color};'
                                f'height:8px;border-radius:4px;"></div></div>',
                                unsafe_allow_html=True,
                            )

                    st.divider()

            # Stats bar
            github_count = stats.get("github_count", 0)
            pdf_count = stats.get("pdf_count", 0)
            st.caption(
                f"⚡ {retrieval_count} chunks retrieved  |  "
                f"🐙 {github_count} from codebase  |  "
                f"📄 {pdf_count} from documents"
            )

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your codebase or documents..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the query endpoint
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            resp = _api_post("/query", json={"question": prompt, "top_k": top_k})

        if resp is None:
            error_text = f"❌ Cannot connect to backend at {API_URL}. Is it running?"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        elif resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            error_text = f"❌ Query failed ({resp.status_code}): {detail}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        else:
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            retrieval_count = data.get("retrieval_count", len(sources))

            github_count = sum(
                1 for s in sources
                if s.get("source_type", s.get("source", "")) == "github"
            )
            pdf_count = sum(
                1 for s in sources
                if s.get("source_type", s.get("source", "")) == "pdf"
            )

            # Display answer
            st.markdown(answer)

            # Display sources
            if show_sources and sources:
                with st.expander(f"📎 {retrieval_count} sources retrieved — click to expand"):
                    for src in sources:
                        source_type = src.get("source_type", src.get("source", "unknown"))
                        score = src.get("similarity_score", src.get("score"))

                        col_content, col_score = st.columns([3, 1])

                        with col_content:
                            if source_type == "github":
                                file_path = src.get("file_path", "unknown")
                                lang = src.get("language", "")
                                color = LANG_COLORS.get(lang, DEFAULT_LANG_COLOR)
                                st.markdown(
                                    f"📄 **{file_path}** "
                                    f'<span style="background-color:{color};color:#fff;'
                                    f'padding:2px 8px;border-radius:4px;font-size:0.8em;">'
                                    f"{lang}</span>",
                                    unsafe_allow_html=True,
                                )
                                preview = src.get("content_preview", "")
                                if preview:
                                    st.code(preview, language=lang or None)
                            else:
                                file_name = src.get("file_name", src.get("file_path", "unknown"))
                                page = src.get("page", src.get("page_number", "?"))
                                st.markdown(f"📕 **{file_name}** — page {page}")
                                preview = src.get("content_preview", "")
                                if preview:
                                    st.markdown(f"> {preview}")

                        with col_score:
                            if show_scores and score is not None:
                                st.metric("Score", f"{score:.2f}")
                                if score >= 0.85:
                                    bar_color = "#2ecc71"
                                elif score >= 0.70:
                                    bar_color = "#f39c12"
                                else:
                                    bar_color = "#e74c3c"
                                st.markdown(
                                    f'<div style="background:#eee;border-radius:4px;height:8px;">'
                                    f'<div style="width:{score * 100:.0f}%;background:{bar_color};'
                                    f'height:8px;border-radius:4px;"></div></div>',
                                    unsafe_allow_html=True,
                                )

                        st.divider()

            # Stats bar
            st.caption(
                f"⚡ {retrieval_count} chunks retrieved  |  "
                f"🐙 {github_count} from codebase  |  "
                f"📄 {pdf_count} from documents"
            )

            # Persist message with sources and stats
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "stats": {
                        "retrieval_count": retrieval_count,
                        "github_count": github_count,
                        "pdf_count": pdf_count,
                    },
                }
            )
