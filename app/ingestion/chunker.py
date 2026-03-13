"""AST-aware code chunker using LangChain's language-specific splitters."""

import logging
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """A single chunk of code with associated metadata."""

    text: str
    file_path: str
    language: str
    chunk_index: int
    start_line: int
    repo_name: str


# Map file extension → LangChain Language enum
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".md": Language.MARKDOWN,
}


def _get_splitter(extension: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware splitter or a generic one as fallback.

    Args:
        extension: File extension including the leading dot.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    lang = EXTENSION_TO_LANGUAGE.get(extension)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    # Fallback for YAML and unsupported languages
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def _estimate_start_line(full_text: str, chunk_text: str) -> int:
    """Estimate the 1-based starting line number of a chunk in the source file.

    Args:
        full_text: Complete file content.
        chunk_text: The chunk substring.

    Returns:
        Estimated 1-based line number.
    """
    idx = full_text.find(chunk_text)
    if idx == -1:
        return 1
    return full_text[:idx].count("\n") + 1


def chunk_code_file(
    content: str,
    file_path: str,
    language: str,
    repo_name: str,
) -> list[CodeChunk]:
    """Split a single code file into overlapping, language-aware chunks.

    Args:
        content: Raw file content.
        file_path: Relative path of the file in the repository.
        language: Human-readable language name.
        repo_name: Name of the source repository.

    Returns:
        List of CodeChunk instances.
    """
    if not content.strip():
        return []

    extension = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    splitter = _get_splitter(extension)
    split_texts = splitter.split_text(content)

    chunks: list[CodeChunk] = []
    for idx, text in enumerate(split_texts):
        chunks.append(
            CodeChunk(
                text=text,
                file_path=file_path,
                language=language,
                chunk_index=idx,
                start_line=_estimate_start_line(content, text),
                repo_name=repo_name,
            )
        )

    logger.debug("Chunked %s → %d chunks", file_path, len(chunks))
    return chunks


def chunk_code_files(
    code_files: list[dict],
) -> list[CodeChunk]:
    """Chunk a batch of code files.

    Args:
        code_files: List of dicts with keys: content, file_path, language, repo_name.

    Returns:
        Combined list of CodeChunk instances.
    """
    all_chunks: list[CodeChunk] = []
    for cf in code_files:
        all_chunks.extend(
            chunk_code_file(
                content=cf["content"],
                file_path=cf["file_path"],
                language=cf["language"],
                repo_name=cf["repo_name"],
            )
        )
    logger.info("Total code chunks: %d from %d files", len(all_chunks), len(code_files))
    return all_chunks
