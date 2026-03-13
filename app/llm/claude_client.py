"""Gemini LLM client for generating answers from retrieved context."""

import logging
from typing import Any

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a code and documentation expert. Answer questions using the "
    "provided context from a GitHub codebase and PDF documents. Always cite "
    "which file or document your answer comes from."
)


def _build_context_string(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a labelled context block for the LLM.

    Args:
        chunks: List of retrieval result dicts.

    Returns:
        Formatted context string.
    """
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


class GeminiClient:
    """Wrapper around the Google Gemini API for question answering."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model_name = settings.gemini_llm_model
        logger.info("Gemini LLM initialised with model '%s'", self._model_name)

    def query(
        self,
        user_question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        """Send a question + retrieved context to Gemini and return the answer.

        Args:
            user_question: The user's natural language question.
            retrieved_chunks: List of retrieval result dicts.

        Returns:
            The model's answer as a string.
        """
        if not retrieved_chunks:
            context_block = "(No relevant context was found.)"
        else:
            context_block = _build_context_string(retrieved_chunks)

        user_message = (
            f"## Retrieved Context\n\n{context_block}\n\n"
            f"## Question\n\n{user_question}"
        )

        logger.info(
            "Sending query to Gemini (%s) with %d context chunks",
            self._model_name,
            len(retrieved_chunks),
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )

        answer = response.text
        logger.info("Received answer (%d chars)", len(answer))
        return answer
