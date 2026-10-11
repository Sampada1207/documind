"""Comprehensive test suite for document upload endpoints and service layer."""

import io
import unittest
from unittest.mock import MagicMock, patch

from app.services.pdf_service import (
    MAX_PDF_SIZE_BYTES,
    EmptyPDFError,
    InvalidPDFError,
    PDFSizeLimitExceededError,
    ScannedPDFError,
    extract_text_from_pdf,
)

# TestClient requires httpx (from requirements.txt)
try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False
    client = None


class TestPDFServiceUnit(unittest.TestCase):
    """Unit tests for the PDF extraction service layer."""

    def test_extract_empty_bytes_raises_empty_pdf_error(self):
        """0-byte input must raise EmptyPDFError."""
        with self.assertRaises(EmptyPDFError) as ctx:
            extract_text_from_pdf(b"", "empty.pdf")
        self.assertIn("Uploaded PDF file is empty", str(ctx.exception))

    def test_extract_oversized_bytes_raises_size_limit_error(self):
        """Bytes exceeding 10 MB limit must raise PDFSizeLimitExceededError."""
        large_payload = b"%PDF-" + b"x" * (MAX_PDF_SIZE_BYTES + 10)
        with self.assertRaises(PDFSizeLimitExceededError) as ctx:
            extract_text_from_pdf(large_payload, "oversized.pdf")
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    def test_extract_non_pdf_header_raises_invalid_pdf_error(self):
        """Bytes without %PDF- magic header must raise InvalidPDFError."""
        with self.assertRaises(InvalidPDFError) as ctx:
            extract_text_from_pdf(b"PlainTextNotAPdf", "test.txt")
        self.assertIn("missing PDF header", str(ctx.exception))

    def test_extract_scanned_pdf_zero_text_raises_scanned_pdf_error(self):
        """PDF with pages but zero extractable text must raise ScannedPDFError with OCR notice."""
        mock_doc = MagicMock()
        mock_doc.page_count = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "   \n\t "
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]

        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            import fitz
            fitz.open.return_value = mock_doc

            with self.assertRaises(ScannedPDFError) as ctx:
                extract_text_from_pdf(b"%PDF-1.5", "scanned.pdf")
            self.assertIn("Scanned PDFs require OCR", str(ctx.exception))
            mock_doc.close.assert_called_once()

    def test_extract_multiple_pages_success(self):
        """Multi-page document extraction preserves 1-based page numbers and computes word metrics."""
        mock_doc = MagicMock()
        mock_doc.page_count = 3

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1: Introduction to Natural Language Processing."
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2: Vector embeddings and FAISS index."
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3: Retrieval-Augmented Generation."
        mock_doc.load_page.side_effect = [mock_page1, mock_page2, mock_page3]

        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            import fitz
            fitz.open.return_value = mock_doc

            result = extract_text_from_pdf(b"%PDF-1.4", "documind_research.pdf")

            self.assertEqual(result["filename"], "documind_research.pdf")
            self.assertEqual(result["total_pages"], 3)
            self.assertEqual(len(result["pages"]), 3)
            self.assertEqual(result["pages"][0]["page"], 1)
            self.assertIn("Introduction", result["pages"][0]["text"])
            self.assertEqual(result["pages"][1]["page"], 2)
            self.assertEqual(result["pages"][2]["page"], 3)
            self.assertGreater(result["total_characters"], 0)
            self.assertGreater(result["total_words"], 0)
            mock_doc.close.assert_called_once()


class TestDocumentUploadEndpoint(unittest.TestCase):
    """Integration tests for POST /api/v1/documents/upload."""

    def setUp(self):
        if not HAS_TESTCLIENT:
            self.skipTest("FastAPI TestClient requires httpx (pip install -r requirements.txt)")

    def test_endpoint_rejects_non_pdf_extension(self):
        """Uploading a non-.pdf file must return 400 Bad Request."""
        file_content = io.BytesIO(b"Hello world")
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", file_content, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files (.pdf) are supported", response.json()["detail"])

    def test_endpoint_rejects_empty_file(self):
        """Uploading an empty 0-byte PDF must return 400 Bad Request."""
        file_content = io.BytesIO(b"")
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", file_content, "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty (0 bytes)", response.json()["detail"])

    def test_endpoint_rejects_oversized_file(self):
        """Uploading a file larger than 10 MB must return 413 Payload Too Large."""
        oversized_bytes = io.BytesIO(b"%PDF-" + b"0" * (MAX_PDF_SIZE_BYTES + 50))
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("large.pdf", oversized_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds maximum allowed size", response.json()["detail"])

    def test_endpoint_rejects_corrupted_pdf(self):
        """Corrupted PDF bytes must return 400 Bad Request."""
        corrupted_bytes = io.BytesIO(b"%PDF-CORRUPTED_GARBAGE_PAYLOAD")
        with patch("app.api.documents.extract_text_from_pdf") as mock_extract:
            mock_extract.side_effect = InvalidPDFError(
                "The uploaded PDF file is malformed, encrypted, or corrupted."
            )
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("corrupt.pdf", corrupted_bytes, "application/pdf")},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("malformed", response.json()["detail"])

    def test_endpoint_rejects_scanned_pdf_with_422(self):
        """Scanned PDF with zero extractable text must return 422 with OCR advice."""
        pdf_bytes = io.BytesIO(b"%PDF-IMAGE_ONLY")
        with patch("app.api.documents.extract_text_from_pdf") as mock_extract:
            mock_extract.side_effect = ScannedPDFError(
                "No extractable text found in this PDF. Scanned PDFs require OCR, which is not implemented yet."
            )
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("scanned_doc.pdf", pdf_bytes, "application/pdf")},
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("OCR", response.json()["detail"])

    def test_endpoint_successful_upload(self):
        """Valid PDF upload must return structured JSON with 200 OK."""
        pdf_bytes = io.BytesIO(b"%PDF-SAMPLE")
        expected_response = {
            "document_id": "test-doc-uuid-1234",
            "filename": "sample.pdf",
            "total_pages": 1,
            "total_characters": 42,
            "total_words": 6,
            "pages": [
                {
                    "page": 1,
                    "text": "Antigravity artificial intelligence research project.",
                    "char_count": 42,
                    "word_count": 6,
                }
            ],
        }

        with patch("app.api.documents.extract_text_from_pdf", return_value=expected_response):
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["document_id"], "test-doc-uuid-1234")
            self.assertEqual(data["filename"], "sample.pdf")
            self.assertEqual(data["total_pages"], 1)
            self.assertEqual(len(data["pages"]), 1)
            self.assertEqual(data["pages"][0]["page"], 1)
            self.assertIn("Antigravity", data["pages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
