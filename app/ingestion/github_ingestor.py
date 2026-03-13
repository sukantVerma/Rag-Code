"""GitHub repository ingestion: clone, walk, filter, and extract code files."""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a single code file extracted from a repository."""

    file_path: str
    content: str
    language: str
    repo_name: str


# Map file extension to a human-readable language name
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _repo_name_from_url(repo_url: str) -> str:
    """Extract the repository name from a GitHub URL."""
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _should_include(file_path: Path) -> bool:
    """Check whether a file should be ingested based on extension and path."""
    # Exclude directories
    for part in file_path.parts:
        if part in settings.excluded_dirs:
            return False

    return file_path.suffix in settings.supported_extensions


def clone_repository(repo_url: str, pat_token: str | None = None) -> tuple[str, str]:
    """Shallow-clone a GitHub repository and return (clone_path, repo_name).

    Args:
        repo_url: HTTPS URL of the GitHub repository.
        pat_token: Optional personal access token for private repos.

    Returns:
        Tuple of (absolute clone path, repository name).
    """
    repo_name = _repo_name_from_url(repo_url)
    clone_dir = Path(settings.repo_clone_path) / repo_name

    # Remove existing clone to ensure a clean state
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    # Inject PAT into URL for private repositories
    clone_url = repo_url
    if pat_token:
        # https://github.com/owner/repo → https://<token>@github.com/owner/repo
        clone_url = repo_url.replace("https://", f"https://{pat_token}@")

    logger.info("Cloning %s (depth=1) into %s", repo_name, clone_dir)
    Repo.clone_from(clone_url, str(clone_dir), depth=1)
    logger.info("Clone complete: %s", repo_name)

    return str(clone_dir), repo_name


def walk_repository(clone_path: str, repo_name: str) -> list[CodeFile]:
    """Walk the cloned repository and return a list of CodeFile objects.

    Args:
        clone_path: Absolute path to the cloned repo.
        repo_name: Name of the repository.

    Returns:
        List of CodeFile dataclass instances.
    """
    code_files: list[CodeFile] = []
    root = Path(clone_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if not _should_include(file_path):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.warning("Could not read file %s — skipping", file_path)
            continue

        relative = str(file_path.relative_to(root))
        language = EXTENSION_LANGUAGE_MAP.get(file_path.suffix, "unknown")

        code_files.append(
            CodeFile(
                file_path=relative,
                content=content,
                language=language,
                repo_name=repo_name,
            )
        )

    logger.info("Collected %d files from %s", len(code_files), repo_name)
    return code_files


def ingest_github_repo(
    repo_url: str, pat_token: str | None = None
) -> list[CodeFile]:
    """Full pipeline: clone a repo and extract all supported code files.

    Args:
        repo_url: HTTPS URL of the GitHub repository.
        pat_token: Optional personal access token.

    Returns:
        List of CodeFile instances.
    """
    settings.ensure_directories()
    clone_path, repo_name = clone_repository(repo_url, pat_token)
    return walk_repository(clone_path, repo_name)


def incremental_ingest(
    repo_url: str,
    changed_files: list[str],
    pat_token: str | None = None,
) -> list[CodeFile]:
    """Ingest only the files that changed (for CI/CD delta updates).

    Args:
        repo_url: HTTPS URL of the GitHub repository.
        changed_files: List of relative file paths that changed.
        pat_token: Optional personal access token.

    Returns:
        List of CodeFile instances for only the changed files.
    """
    settings.ensure_directories()
    repo_name = _repo_name_from_url(repo_url)
    clone_dir = Path(settings.repo_clone_path) / repo_name

    # If the repo isn't already cloned, do a fresh clone
    if not clone_dir.exists():
        clone_path, repo_name = clone_repository(repo_url, pat_token)
    else:
        clone_path = str(clone_dir)

    code_files: list[CodeFile] = []
    root = Path(clone_path)

    for rel_path in changed_files:
        file_path = root / rel_path
        if not file_path.is_file():
            logger.info("Changed file no longer exists (deleted?): %s", rel_path)
            continue
        if not _should_include(file_path):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.warning("Could not read changed file %s — skipping", rel_path)
            continue

        language = EXTENSION_LANGUAGE_MAP.get(file_path.suffix, "unknown")
        code_files.append(
            CodeFile(
                file_path=rel_path,
                content=content,
                language=language,
                repo_name=repo_name,
            )
        )

    logger.info(
        "Incremental ingest: %d changed files processed for %s",
        len(code_files),
        repo_name,
    )
    return code_files
