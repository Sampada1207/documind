"""Unit and integration tests for Phase 5: Document Intelligence (Summarization and Comparison).

Verifies:
1. Document summarization generates structured output (overview, objectives, key_points, findings, conclusion).
2. Preserves source page references for findings.
3. Long documents are safely budgeted without oversized requests.
4. Document comparison analyzes objectives, methodologies, findings, limitations, similarities, and differences.
5. Comparison citations preserve document_id, filename, and page numbers.
6. Error handling: 404 for missing documents, 400 for self-comparison.
7. Mock mode makes zero OpenAI API calls and requires no API key.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services.document_registry import registry
from app.services.summary_service import summary_service, prepare_budgeted_document_context
from app.services.comparison_service import comparison_service
from app.services.openai_service import openai_service

try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False
    client = None


class TestDocumentSummarization(unittest.TestCase):
    """Tests for document summarization service and context budgeting."""

    def setUp(self):
        registry._documents.clear()
        registry._indices.clear()

    def test_summarize_document_structured_output_with_page_citations(self):
        """Summarization extracts structured fields and attaches page citations."""
        doc_id = "doc_summary_test_1"
        registry.register_document({
            "document_id": doc_id,
            "filename": "solar_pilot_spec.pdf",
            "total_pages": 3,
            "total_characters": 1500,
            "total_words": 200,
            "pages": [
                {
                    "page": 1,
                    "text": "The Riverside Solar Pilot demonstrates municipal rooftop solar deployment. Project Objectives: Deploy solar across 6 municipal buildings.",
                    "char_count": 130,
                    "word_count": 18,
                },
                {
                    "page": 2,
                    "text": "Technical Findings: Total planned solar capacity is 240 kW. Monocrystalline modules deliver 21.5% average efficiency.",
                    "char_count": 115,
                    "word_count": 14,
                },
                {
                    "page": 3,
                    "text": "Financial Summary: Total project budget is ₹1.8 crore. In conclusion, the project provides a scalable template for municipal microgrids.",
                    "char_count": 135,
                    "word_count": 17,
                },
            ],
        })

        with patch.object(settings, "LLM_PROVIDER", "mock"):
            summary = summary_service.summarize_document(doc_id)

        self.assertEqual(summary["document_id"], doc_id)
        self.assertEqual(summary["filename"], "solar_pilot_spec.pdf")
        self.assertEqual(summary["total_pages"], 3)
        self.assertTrue(summary["is_simulated"])
        self.assertEqual(summary["llm_provider"], "mock")
        self.assertTrue(len(summary["overview"]) > 10)
        self.assertTrue(len(summary["conclusion"]) > 10)

        # Verify page references in findings
        self.assertTrue(len(summary["findings"]) > 0)
        first_finding = summary["findings"][0]
        self.assertIn("text", first_finding)
        self.assertIn("page", first_finding)
        self.assertIn(first_finding["page"], [1, 2, 3])

    def test_long_document_context_budgeting_safely_samples_pages(self):
        """Documents with many pages are safely sampled within the character budget."""
        long_doc_id = "doc_long_50_pages"
        pages = []
        for i in range(1, 25):
            pages.append({
                "page": i,
                "text": f"Page {i} content. Extensive analysis and tabular metrics for section {i}." * 20,
                "char_count": 1000,
                "word_count": 150,
            })

        doc_data = {
            "document_id": long_doc_id,
            "filename": "massive_report.pdf",
            "total_pages": 24,
            "total_characters": 24000,
            "total_words": 3600,
            "pages": pages,
        }
        registry.register_document(doc_data)

        budgeted_text, sampled_pages = prepare_budgeted_document_context(doc_data)
        # Verify budgeted text stays safely below 13,000 characters
        self.assertLess(len(budgeted_text), 13000)
        # Sampled pages should include page 1 and page 24
        sampled_numbers = [p["page"] for p in sampled_pages]
        self.assertIn(1, sampled_numbers)
        self.assertIn(24, sampled_numbers)

    def test_summarize_missing_document_raises_keyerror(self):
        """Attempting to summarize a non-existent document raises KeyError."""
        with self.assertRaises(KeyError) as ctx:
            summary_service.summarize_document("unknown_doc_id")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_summarize_mock_mode_makes_zero_openai_calls(self):
        """Mock summarization never invokes OpenAI."""
        doc_id = "doc_zero_calls_summary"
        registry.register_document({
            "document_id": doc_id,
            "filename": "test.pdf",
            "total_pages": 1,
            "total_characters": 200,
            "total_words": 30,
            "pages": [{"page": 1, "text": "This is a simple document with basic findings.", "char_count": 45, "word_count": 8}],
        })

        with patch.object(settings, "LLM_PROVIDER", "mock"), \
             patch.object(settings, "OPENAI_API_KEY", None), \
             patch.object(openai_service, "generate_rag_answer", side_effect=AssertionError("OpenAI must not be called")):
            
            result = summary_service.summarize_document(doc_id)

        self.assertEqual(result["llm_provider"], "mock")
        self.assertTrue(result["is_simulated"])


class TestDocumentComparison(unittest.TestCase):
    """Tests for research paper and document comparison service."""

    def setUp(self):
        registry._documents.clear()
        registry._indices.clear()

        self.doc_a_id = "doc_solar_riverside"
        registry.register_document({
            "document_id": self.doc_a_id,
            "filename": "riverside_solar.pdf",
            "total_pages": 2,
            "total_characters": 800,
            "total_words": 110,
            "pages": [
                {
                    "page": 1,
                    "text": "Objectives: Deploy 240 kW rooftop solar across 6 municipal buildings in Riverside.",
                    "char_count": 80,
                    "word_count": 11,
                },
                {
                    "page": 2,
                    "text": "Methodology utilizes monocrystalline PV panels. Results show 22% efficiency. Limitation: Grid interconnection delays remain a bottleneck.",
                    "char_count": 140,
                    "word_count": 16,
                },
            ],
        })

        self.doc_b_id = "doc_wind_coastal"
        registry.register_document({
            "document_id": self.doc_b_id,
            "filename": "coastal_wind.pdf",
            "total_pages": 2,
            "total_characters": 850,
            "total_words": 120,
            "pages": [
                {
                    "page": 1,
                    "text": "Objectives: Evaluate 500 kW coastal wind turbines for decentralized grid backup.",
                    "char_count": 80,
                    "word_count": 10,
                },
                {
                    "page": 2,
                    "text": "Approach uses horizontal axis turbines. Results show 34% capacity factor. Challenge: Salt corrosion requires frequent coating inspections.",
                    "char_count": 145,
                    "word_count": 17,
                },
            ],
        })

    def test_compare_two_documents_success_with_citations(self):
        """Comparing two distinct documents returns structured comparison with source provenance."""
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            comparison = comparison_service.compare_documents(self.doc_a_id, self.doc_b_id)

        self.assertEqual(comparison["document_a"]["document_id"], self.doc_a_id)
        self.assertEqual(comparison["document_b"]["document_id"], self.doc_b_id)
        self.assertEqual(comparison["document_a"]["filename"], "riverside_solar.pdf")
        self.assertEqual(comparison["document_b"]["filename"], "coastal_wind.pdf")

        # Verify similarities, differences, missing info
        self.assertTrue(len(comparison["similarities"]) > 0)
        self.assertTrue(len(comparison["differences"]) > 0)
        self.assertTrue(len(comparison["missing_information"]) > 0)

        # Verify citations attach document IDs and filenames
        self.assertTrue(len(comparison["citations"]) > 0)
        cit = comparison["citations"][0]
        self.assertIn("document_id", cit)
        self.assertIn("filename", cit)
        self.assertIn("page", cit)
        self.assertIn("excerpt", cit)

    def test_compare_self_raises_value_error(self):
        """Comparing a document to itself raises ValueError (400 Bad Request)."""
        with self.assertRaises(ValueError) as ctx:
            comparison_service.compare_documents(self.doc_a_id, self.doc_a_id)
        self.assertIn("cannot compare a document with itself", str(ctx.exception).lower())

    def test_compare_missing_document_raises_key_error(self):
        """Comparing with a missing document ID raises KeyError (404 Not Found)."""
        with self.assertRaises(KeyError) as ctx:
            comparison_service.compare_documents(self.doc_a_id, "non_existent_b")
        self.assertIn("not found", str(ctx.exception).lower())


class TestPhase5HTTPEndpoints(unittest.TestCase):
    """End-to-end HTTP endpoint verification for Phase 5."""

    def setUp(self):
        registry._documents.clear()
        registry._indices.clear()

        self.doc_1_id = "http_doc_1"
        registry.register_document({
            "document_id": self.doc_1_id,
            "filename": "paper_alpha.pdf",
            "total_pages": 2,
            "total_characters": 600,
            "total_words": 90,
            "pages": [
                {"page": 1, "text": "Paper Alpha proposes neural attention networks. Objectives: Improve translation BLEU score.", "char_count": 90, "word_count": 12},
                {"page": 2, "text": "Findings: Achieved 28.4 BLEU. Conclusion: Attention mechanisms replace recurrent connections.", "char_count": 92, "word_count": 10},
            ],
        })

        self.doc_2_id = "http_doc_2"
        registry.register_document({
            "document_id": self.doc_2_id,
            "filename": "paper_beta.pdf",
            "total_pages": 2,
            "total_characters": 650,
            "total_words": 95,
            "pages": [
                {"page": 1, "text": "Paper Beta evaluates recurrent LSTM architectures. Aim: Model long-range sequences.", "char_count": 85, "word_count": 11},
                {"page": 2, "text": "Results: 25.1 BLEU. Limitations: Training latency scales linearly with sequence length.", "char_count": 88, "word_count": 10},
            ],
        })

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_list_all_documents_endpoint(self):
        """GET /api/v1/documents returns list of registered documents."""
        response = client.get("/api/v1/documents")
        self.assertEqual(response.status_code, 200)
        docs = response.json()
        self.assertEqual(len(docs), 2)
        doc_ids = [d["document_id"] for d in docs]
        self.assertIn(self.doc_1_id, doc_ids)
        self.assertIn(self.doc_2_id, doc_ids)

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_post_summarize_endpoint_success(self):
        """POST /api/v1/documents/{doc_id}/summarize returns 200 with structured summary."""
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            response = client.post(f"/api/v1/documents/{self.doc_1_id}/summarize")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_id"], self.doc_1_id)
        self.assertEqual(data["filename"], "paper_alpha.pdf")
        self.assertTrue(data["is_simulated"])
        self.assertIn("overview", data)
        self.assertIn("findings", data)
        self.assertIn("conclusion", data)

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_post_summarize_missing_doc_returns_404(self):
        """POST /api/v1/documents/missing/summarize returns 404."""
        response = client.post("/api/v1/documents/non_existent_doc/summarize")
        self.assertEqual(response.status_code, 404)

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_post_compare_endpoint_success(self):
        """POST /api/v1/documents/compare returns 200 with comparison analysis."""
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            response = client.post(
                "/api/v1/documents/compare",
                json={"document_id_a": self.doc_1_id, "document_id_b": self.doc_2_id},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_a"]["document_id"], self.doc_1_id)
        self.assertEqual(data["document_b"]["document_id"], self.doc_2_id)
        self.assertTrue(len(data["similarities"]) > 0)
        self.assertTrue(len(data["differences"]) > 0)

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_post_compare_same_doc_returns_400(self):
        """POST /api/v1/documents/compare with identical doc IDs returns 400 Bad Request."""
        response = client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": self.doc_1_id, "document_id_b": self.doc_1_id},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot compare a document with itself", response.json()["detail"].lower())

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_post_compare_missing_doc_returns_404(self):
        """POST /api/v1/documents/compare with missing doc returns 404."""
        response = client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": self.doc_1_id, "document_id_b": "missing_doc"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
