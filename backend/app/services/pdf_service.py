"""In-memory PDF text extraction and validation service using PyMuPDF (fitz)."""

import uuid
from typing import Any, Dict, List

# 10 MB maximum upload limit in bytes
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class InvalidPDFError(PDFProcessingError):
    """Raised when the file is not a valid PDF or is corrupted."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when the PDF file contains no bytes or no pages."""
    pass


class ScannedPDFError(PDFProcessingError):
    """Raised when a PDF contains no extractable text (e.g. scanned image-only PDF)."""
    pass


class PDFSizeLimitExceededError(PDFProcessingError):
    """Raised when the PDF exceeds the maximum size limit."""
    pass


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract text from a PDF file in-memory page by page.
    
    Args:
        file_bytes: Raw binary content of the PDF.
        filename: Original filename of the uploaded PDF.
        
    Returns:
        Structured dict with document_id, filename, total_pages, and pages list.
        
    Raises:
        EmptyPDFError: If the file is 0 bytes or has 0 pages.
        PDFSizeLimitExceededError: If the file exceeds 10 MB.
        InvalidPDFError: If the file cannot be opened as a PDF.
        ScannedPDFError: If no extractable text is present in the document.
    """
    if not file_bytes:
        raise EmptyPDFError("Uploaded PDF file is empty.")

    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        raise PDFSizeLimitExceededError(
            f"File exceeds maximum allowed size of {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # Basic PDF magic header validation (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        raise InvalidPDFError("File does not appear to be a valid PDF document (missing PDF header).")

    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF is not installed. Please install pymupdf to process PDFs.") from e

    doc = None
    try:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise InvalidPDFError("The uploaded PDF file is malformed, encrypted, or corrupted.") from e

        if doc.page_count == 0:
            raise EmptyPDFError("The PDF document contains 0 pages.")

        pages: List[Dict[str, Any]] = []
        total_characters = 0
        total_words = 0

        for page_idx in range(doc.page_count):
            page_number = page_idx + 1  # 1-indexed page numbering
            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text() or ""
            except Exception as e:
                raise InvalidPDFError(f"Failed to read page {page_number} from the PDF document.") from e

            # Clean and compute metrics
            cleaned_text = raw_text.strip()
            char_count = len(cleaned_text)
            words = cleaned_text.split() if cleaned_text else []
            word_count = len(words)

            total_characters += char_count
            total_words += word_count

            pages.append({
                "page": page_number,
                "text": raw_text,
                "char_count": char_count,
                "word_count": word_count,
            })

        # Reject PDFs that contain no extractable text across all pages
        if total_characters == 0:
            raise ScannedPDFError(
                "No extractable text found in this PDF. Scanned PDFs require OCR, which is not implemented yet."
            )

        return {
            "document_id": str(uuid.uuid4()),
            "filename": filename,
            "total_pages": doc.page_count,
            "total_characters": total_characters,
            "total_words": total_words,
            "pages": pages,
        }

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
