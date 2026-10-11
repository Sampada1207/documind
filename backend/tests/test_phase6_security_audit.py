"""Phase 6.1 Security and Configuration Audit Regression Tests.

Validates:
1. Secret hygiene (.env.example has zero active secrets).
2. Document ID validation prevents malformed / path-traversal IDs.
3. PDF upload filename sanitization strips directory traversal & dangerous characters.
4. DocumentRegistry capacity is strictly bounded (DoS mitigation).
5. DELETE /api/v1/documents/{document_id} removes documents and vector indices.
6. OpenAI rate limits (429) and network timeouts (504) are properly mapped on /summarize and /compare.
7. Mock LLM mode functions completely offline with zero OpenAI API calls.
"""

import io
import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.document_registry import DocumentRegistry, registry
from app.services.openai_service import (
    OpenAIRateLimitError,
    OpenAINetworkError,
    OpenAIAuthenticationError,
    openai_service,
)


class TestSecurityAndConfigurationAudit(unittest.TestCase):
    """Regression tests for Phase 6.1 security and configuration audit."""

    def setUp(self):
        self.client = TestClient(app)
        registry.clear()
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        registry.clear()

    def test_env_example_has_no_active_secrets(self):
        """Verify backend/.env.example contains no real OpenAI API keys or credentials."""
        env_example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        self.assertTrue(os.path.exists(env_example_path), ".env.example must exist")

        with open(env_example_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Must not contain sk-proj- or active OpenAI key prefix
        self.assertNotIn("sk-proj-", content, "Active OpenAI project key detected in .env.example!")
        self.assertNotIn("sk-ant-", content)
        # Should have placeholder
        self.assertIn("your-openai-api-key-here", content)

    def test_document_id_validation_rejects_malformed_ids(self):
        """Endpoints must reject malformed document IDs and injection attempts with HTTP 400."""
        # 1. Path parameter validation (characters that fit in a single path segment but violate ID format)
        bad_path_ids = [
            "doc<script>",
            "doc;drop_table",
            "doc$injection",
            "a" * 150,  # exceeds 128 chars limit
        ]

        for bad_id in bad_path_ids:
            for route_fn, method in [
                (lambda id: self.client.post(f"/api/v1/documents/{id}/index"), "POST index"),
                (lambda id: self.client.post(f"/api/v1/documents/{id}/search", json={"query": "test"}), "POST search"),
                (lambda id: self.client.post(f"/api/v1/documents/{id}/ask", json={"question": "test"}), "POST ask"),
                (lambda id: self.client.post(f"/api/v1/documents/{id}/summarize"), "POST summarize"),
                (lambda id: self.client.delete(f"/api/v1/documents/{id}"), "DELETE document"),
            ]:
                res = route_fn(bad_id)
                self.assertEqual(
                    res.status_code, 400,
                    f"Expected 400 for bad id '{bad_id}' on {method}, got {res.status_code}"
                )

        # 2. JSON body validation in /compare (testing raw traversal strings inside payload)
        body_injection_ids = [
            "../../etc/passwd",
            "doc/sub/path",
            "doc\x00null",
            "doc<script>alert(1)</script>",
        ]
        for bad_id in body_injection_ids:
            res = self.client.post(
                "/api/v1/documents/compare",
                json={"document_id_a": bad_id, "document_id_b": "valid_doc_b"},
            )
            self.assertEqual(
                res.status_code, 400,
                f"Expected 400 for traversal id '{bad_id}' in compare payload, got {res.status_code}"
            )


    def test_pdf_upload_sanitizes_filename(self):
        """PDF upload must strip path traversal characters from uploaded filename."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Safe test document content.")
        pdf_bytes = doc.tobytes()
        doc.close()

        # Upload with path traversal and HTML tags in filename
        malicious_name = "../../exploit<script>.pdf"
        response = self.client.post(
            "/api/v1/documents/upload",
            files={"file": (malicious_name, io.BytesIO(pdf_bytes), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        saved_filename = data["filename"]

        # Verify no directory traversal or HTML tags
        self.assertNotIn("..", saved_filename)
        self.assertNotIn("/", saved_filename)
        self.assertNotIn("<", saved_filename)
        self.assertNotIn(">", saved_filename)
        self.assertTrue(saved_filename.endswith(".pdf"))

    def test_document_registry_bounded_capacity(self):
        """DocumentRegistry must enforce maximum capacity and evict oldest documents (DoS protection)."""
        small_registry = DocumentRegistry(max_capacity=3)

        # Register 3 documents
        for i in range(1, 4):
            small_registry.register_document({
                "document_id": f"doc_{i}",
                "filename": f"file_{i}.pdf",
                "total_pages": 1,
                "total_characters": 100,
                "total_words": 20,
                "pages": [{"page": 1, "text": f"Content {i}"}],
            })

        self.assertEqual(len(small_registry._documents), 3)
        self.assertIsNotNone(small_registry.get_document("doc_1"))

        # Register 4th document -> doc_1 should be evicted (FIFO)
        small_registry.register_document({
            "document_id": "doc_4",
            "filename": "file_4.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 20,
            "pages": [{"page": 1, "text": "Content 4"}],
        })

        self.assertEqual(len(small_registry._documents), 3)
        self.assertIsNone(small_registry.get_document("doc_1"), "Oldest doc_1 should have been evicted")
        self.assertIsNotNone(small_registry.get_document("doc_2"))
        self.assertIsNotNone(small_registry.get_document("doc_4"))

    def test_delete_document_endpoint_removes_from_memory(self):
        """DELETE /api/v1/documents/{document_id} removes document and returns 200."""
        registry.register_document({
            "document_id": "doc_to_delete",
            "filename": "delete_me.pdf",
            "total_pages": 1,
            "total_characters": 50,
            "total_words": 10,
            "pages": [{"page": 1, "text": "Temporary content"}],
        })
        self.assertIsNotNone(registry.get_document("doc_to_delete"))

        # Delete existing document
        res = self.client.delete("/api/v1/documents/doc_to_delete")
        self.assertEqual(res.status_code, 200)
        self.assertIn("successfully removed", res.json()["message"])
        self.assertIsNone(registry.get_document("doc_to_delete"))

        # Deleting again returns 404
        res2 = self.client.delete("/api/v1/documents/doc_to_delete")
        self.assertEqual(res2.status_code, 404)

    @patch("app.services.summary_service.summary_service.summarize_document")
    def test_summarize_maps_rate_limit_and_network_errors(self, mock_summarize):
        """Summarize endpoint must map OpenAIRateLimitError to 429 and OpenAINetworkError to 504."""
        registry.register_document({
            "document_id": "doc_err_test",
            "filename": "test.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 20,
            "pages": [{"page": 1, "text": "Text"}],
        })

        # 429 Rate Limit
        mock_summarize.side_effect = OpenAIRateLimitError("Rate limit exceeded")
        res = self.client.post("/api/v1/documents/doc_err_test/summarize")
        self.assertEqual(res.status_code, 429)

        # 504 Gateway Timeout
        mock_summarize.side_effect = OpenAINetworkError("Connection timed out")
        res = self.client.post("/api/v1/documents/doc_err_test/summarize")
        self.assertEqual(res.status_code, 504)

    @patch("app.services.comparison_service.comparison_service.compare_documents")
    def test_compare_maps_rate_limit_and_network_errors(self, mock_compare):
        """Compare endpoint must map OpenAIRateLimitError to 429 and OpenAINetworkError to 504."""
        # 429 Rate Limit
        mock_compare.side_effect = OpenAIRateLimitError("Rate limit exceeded")
        res = self.client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": "doc_a", "document_id_b": "doc_b"},
        )
        self.assertEqual(res.status_code, 429)

        # 504 Gateway Timeout
        mock_compare.side_effect = OpenAINetworkError("Connection timed out")
        res = self.client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": "doc_a", "document_id_b": "doc_b"},
        )
        self.assertEqual(res.status_code, 504)

    def test_mock_mode_zero_api_calls_in_full_workflow(self):
        """Ensure mock mode executes end-to-end without calling OpenAI API."""
        registry.register_document({
            "document_id": "doc_mock_safe",
            "filename": "safe_pilot.pdf",
            "total_pages": 2,
            "total_characters": 400,
            "total_words": 70,
            "pages": [
                {"page": 1, "text": "Objective: Deploy 50 MW solar installation. Results: Achieved 98% efficiency."},
                {"page": 2, "text": "Limitations: Nighttime battery storage constraint. Conclusion: Scalable system."},
            ],
        })

        with patch.object(openai_service, "generate_rag_answer") as mock_openai_call:
            # 1. Summarize
            sum_res = self.client.post("/api/v1/documents/doc_mock_safe/summarize")
            self.assertEqual(sum_res.status_code, 200)
            self.assertTrue(sum_res.json()["is_simulated"])

            # 2. Compare (with another registered document)
            registry.register_document({
                "document_id": "doc_mock_safe_2",
                "filename": "second_study.pdf",
                "total_pages": 1,
                "total_characters": 200,
                "total_words": 30,
                "pages": [{"page": 1, "text": "Methodology: Wind turbines analyzed. Results: 40 MW capacity."}],
            })
            cmp_res = self.client.post(
                "/api/v1/documents/compare",
                json={"document_id_a": "doc_mock_safe", "document_id_b": "doc_mock_safe_2"},
            )
            self.assertEqual(cmp_res.status_code, 200)
            self.assertTrue(cmp_res.json()["is_simulated"])

            # Verify zero OpenAI API calls occurred
            mock_openai_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
