"""PDF document ingestion: extract text and chunk by section."""

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """A single chunk of text extracted from a PDF document."""

    text: str
    page_number: int
    source_file: str
    chunk_index: int


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF file page by page using PyMuPDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts with keys 'text' and 'page_number'.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict] = []
    doc = fitz.open(str(path))

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages.append({"text": text, "page_number": page_num + 1})
    finally:
        doc.close()

    logger.info("Extracted text from %d pages in %s", len(pages), path.name)
    return pages


def chunk_pdf(pdf_path: str) -> list[PDFChunk]:
    """Parse a PDF and split its text into overlapping chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PDFChunk instances.
    """
    pages = extract_text_from_pdf(pdf_path)
    source_file = Path(pdf_path).name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[PDFChunk] = []
    chunk_index = 0

    for page_info in pages:
        page_text = page_info["text"]
        page_number = page_info["page_number"]
        split_texts = splitter.split_text(page_text)

        for text in split_texts:
            chunks.append(
                PDFChunk(
                    text=text,
                    page_number=page_number,
                    source_file=source_file,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    logger.info(
        "PDF %s: %d pages → %d chunks", source_file, len(pages), len(chunks)
    )
    return chunks


def ingest_all_pdfs(pdf_dir: str | None = None) -> list[PDFChunk]:
    """Ingest every PDF in the configured (or given) directory.

    Args:
        pdf_dir: Optional override for the PDF directory path.

    Returns:
        Combined list of PDFChunk instances from all PDFs.
    """
    directory = Path(pdf_dir or settings.pdf_data_path)
    if not directory.exists():
        logger.warning("PDF directory does not exist: %s", directory)
        return []

    all_chunks: list[PDFChunk] = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        try:
            all_chunks.extend(chunk_pdf(str(pdf_file)))
        except Exception:
            logger.exception("Failed to ingest PDF: %s", pdf_file)

    logger.info("Total PDF chunks ingested: %d", len(all_chunks))
    return all_chunks
