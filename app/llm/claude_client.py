"""Claude LLM client for generating answers from retrieved context."""

import logging
from typing import Any

import anthropic

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a code and documentation expert. Answer questions using the "
    "provided context from a GitHub codebase and PDF documents. Always cite "
    "which file or document your answer comes from."
)


def _build_context_string(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a labelled context block for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        meta = chunk.get("metadata", {})
        document = chunk.get("document", "")

        if source == "github":
            label = f"[Source {i}: GitHub — {meta.get('file_path', 'unknown')}]"
        else:
            label = (
                f"[Source {i}: PDF — {meta.get('source_file', 'unknown')} "
                f"(page {meta.get('page_number', '?')})]"
            )

        parts.append(f"{label}\n{document}")

    return "\n\n---\n\n".join(parts)


class ClaudeClient:
    """Wrapper around the Anthropic Claude API for question answering."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model_name = settings.anthropic_model
        logger.info("Claude LLM initialised with model '%s'", self._model_name)

    def query(
        self,
        user_question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        """Send a question + retrieved context to Claude and return the answer."""
        if not retrieved_chunks:
            context_block = "(No relevant context was found.)"
        else:
            context_block = _build_context_string(retrieved_chunks)

        user_message = (
            f"## Retrieved Context\n\n{context_block}\n\n"
            f"## Question\n\n{user_question}"
        )

        logger.info(
            "Sending query to Claude (%s) with %d context chunks",
            self._model_name,
            len(retrieved_chunks),
        )

        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        logger.info("Received answer (%d chars)", len(answer))
        return answer
