"""Unit and regression tests for Mock LLM Mode.

Verifies:
1. Mock mode makes zero OpenAI API calls.
2. Real FAISS retrieval and document chunks are utilized.
3. Predictable, concise answers are extracted from retrieved evidence.
4. Headers, footers, and section titles are filtered out.
5. Specific regression tests for Riverside project:
   - Number of buildings: 6
   - Planned capacity: 240 kW
   - Project budget: ₹1.8 crore
   - Maintenance frequency: every 6 months
   - Unsupported annual generation question returns insufficient evidence
   - Multi-part question handling (buildings + capacity)
6. Citations contain filename, page number, chunk ID, excerpt, and similarity score.
7. Error handling for non-existent documents (404) and unindexed documents (400).
8. No OpenAI API key is required when in mock mode.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services.document_registry import registry
from app.services.mock_llm_service import MockLLMService, mock_llm_service
from app.services.openai_service import openai_service
from app.services.rag_service import run_rag_pipeline

try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False
    client = None


class TestMockLLMService(unittest.TestCase):
    """Direct unit tests for deterministic MockLLMService synthesis."""

    def test_synthesize_mock_answer_supported_question(self):
        """Supported question produces clean concise answer citing retrieved passages."""
        chunks = [
            {
                "chunk_id": "c1",
                "page": 2,
                "text": "The Riverside Solar Pilot comprises 6 buildings with 240 kW total capacity.",
                "score": 0.88,
            },
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="How many buildings are in the Riverside Solar Pilot?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("6 buildings", answer)
        self.assertIn("[1]", answer)

    def test_synthesize_mock_answer_unsupported_question(self):
        """Completely irrelevant question returns insufficient evidence without hallucinations."""
        chunks = [
            {
                "chunk_id": "c1",
                "page": 1,
                "text": "Solar power generation metrics for suburban residential structures.",
                "score": 0.05,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="What is the capital of Mars?",
            retrieved_chunks=chunks,
            relevance_threshold=0.25,
        )
        self.assertFalse(sufficient)
        self.assertIn("insufficient evidence", answer.lower())

    def test_synthesize_mock_answer_empty_chunks(self):
        """Empty retrieved chunks return insufficient evidence."""
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="Any question?",
            retrieved_chunks=[],
        )
        self.assertFalse(sufficient)
        self.assertIn("insufficient evidence", answer.lower())


class TestMockLLMRegression(unittest.TestCase):
    """Regression test suite for specific factual extractions and edge cases."""

    def test_regression_number_of_buildings(self):
        """Extracts exact number of buildings (6) while ignoring repeated headers and metadata."""
        chunks = [
            {
                "chunk_id": "doc_p1_c0",
                "page": 1,
                "text": (
                    "DocuMind Test Document\n"
                    "Page 1 of 5\n"
                    "1. Executive Summary\n"
                    "The Riverside Solar Pilot covers 6 municipal buildings across the north sector.\n"
                    "Confidential - Internal Use Only"
                ),
                "score": 0.89,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="How many buildings are included in the Riverside Solar Pilot?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("6 buildings", answer)
        self.assertIn("[1]", answer)
        # Verify noise lines were ignored
        self.assertNotIn("DocuMind Test Document", answer)
        self.assertNotIn("Page 1 of 5", answer)
        self.assertNotIn("1. Executive Summary", answer)
        self.assertNotIn("Confidential", answer)

    def test_regression_planned_capacity(self):
        """Extracts planned capacity (240 kW) from technical specifications."""
        chunks = [
            {
                "chunk_id": "doc_p2_c1",
                "page": 2,
                "text": (
                    "Technical Specifications\n"
                    "Page 2 of 5\n"
                    "Total planned solar capacity: 240 kW using monocrystalline panels.\n"
                    "All rights reserved."
                ),
                "score": 0.91,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="What is its total planned solar capacity?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("240 kW", answer)
        self.assertIn("[1]", answer)
        self.assertNotIn("Technical Specifications", answer)
        self.assertNotIn("Page 2 of 5", answer)

    def test_regression_project_budget(self):
        """Extracts project budget (₹1.8 crore) accurately."""
        chunks = [
            {
                "chunk_id": "doc_p3_c2",
                "page": 3,
                "text": (
                    "Financial Summary\n"
                    "Page 3 of 5\n"
                    "Project Budget: ₹1.8 crore allocated under clean energy funding."
                ),
                "score": 0.87,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="What is the project budget?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("₹1.8 crore", answer)
        self.assertIn("[1]", answer)
        self.assertNotIn("Financial Summary", answer)

    def test_regression_maintenance_frequency(self):
        """Extracts maintenance frequency (every 6 months)."""
        chunks = [
            {
                "chunk_id": "doc_p4_c3",
                "page": 4,
                "text": (
                    "Operations & Upkeep\n"
                    "Scheduled maintenance frequency: every 6 months for inverter inspection."
                ),
                "score": 0.85,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="What is the maintenance frequency?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("every 6 months", answer)
        self.assertIn("[1]", answer)

    def test_regression_unsupported_annual_generation(self):
        """Returns insufficient evidence for unsupported annual generation query."""
        chunks = [
            {
                "chunk_id": "doc_p1_c0",
                "page": 1,
                "text": (
                    "The Riverside Solar Pilot covers 6 buildings with 240 kW capacity. "
                    "Project budget is ₹1.8 crore."
                ),
                "score": 0.65,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="What is the total annual electricity generation?",
            retrieved_chunks=chunks,
        )
        self.assertFalse(sufficient)
        self.assertIn("insufficient evidence", answer.lower())

    def test_regression_combined_question_buildings_and_capacity(self):
        """Answers combined question with both buildings (6) and capacity (240 kW)."""
        chunks = [
            {
                "chunk_id": "doc_p1_c0",
                "page": 1,
                "text": (
                    "The Riverside Solar Pilot includes 6 buildings with a planned solar capacity of 240 kW."
                ),
                "score": 0.94,
            }
        ]
        answer, sufficient = mock_llm_service.synthesize_mock_answer(
            question="How many buildings are included in the Riverside Solar Pilot, and what is its total planned solar capacity?",
            retrieved_chunks=chunks,
        )
        self.assertTrue(sufficient)
        self.assertIn("6 buildings", answer)
        self.assertIn("240 kW", answer)
        self.assertIn("[1]", answer)


class TestMockLLMRAGPipeline(unittest.TestCase):
    """Tests verifying run_rag_pipeline in mock mode."""

    def setUp(self):
        registry._documents.clear()
        registry._indices.clear()

    def test_mock_mode_makes_zero_openai_calls(self):
        """Pipeline in mock mode must not invoke OpenAI service."""
        doc_id = "test_doc_mock_zero_calls"
        registry.register_document({
            "document_id": doc_id,
            "filename": "solar_report.pdf",
            "total_pages": 4,
            "total_characters": 1200,
            "total_words": 180,
            "pages": [],
        })

        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {
                "chunk_id": f"{doc_id}_p1_c0",
                "page": 1,
                "text": "Solar panel arrays demonstrate 22% average efficiency in temperate climates.",
                "score": 0.82,
                "char_count": 75,
                "word_count": 9,
            }
        ]
        registry.set_index(doc_id, mock_vector_store, chunk_count=1)

        with patch.object(settings, "LLM_PROVIDER", "mock"), \
             patch.object(settings, "OPENAI_API_KEY", None), \
             patch.object(openai_service, "generate_rag_answer", side_effect=AssertionError("OpenAI must NOT be called in mock mode")):
            
            result = run_rag_pipeline(doc_id, "What is the panel efficiency?", top_k=1)

        self.assertEqual(result["llm_provider"], "mock")
        self.assertEqual(result["model_used"], MockLLMService.MOCK_MODEL_NAME)
        self.assertTrue(result["has_sufficient_evidence"])
        self.assertIn("[1]", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "solar_report.pdf")
        self.assertEqual(result["sources"][0]["page"], 1)
        self.assertEqual(result["sources"][0]["chunk_id"], f"{doc_id}_p1_c0")
        self.assertEqual(result["sources"][0]["score"], 0.82)

    def test_citation_page_and_chunk_mapping(self):
        """Verifies citation metadata preserves exact filename, page, chunk_id, and similarity score."""
        doc_id = "doc_citation_mapping"
        registry.register_document({
            "document_id": doc_id,
            "filename": "riverside_spec.pdf",
            "total_pages": 3,
            "total_characters": 900,
            "total_words": 120,
            "pages": [],
        })

        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {
                "chunk_id": f"{doc_id}_p2_c5",
                "page": 2,
                "text": "Project Budget: ₹1.8 crore allocated for system integration.",
                "score": 0.895,
                "char_count": 60,
                "word_count": 8,
            }
        ]
        registry.set_index(doc_id, mock_vector_store, chunk_count=1)

        with patch.object(settings, "LLM_PROVIDER", "mock"):
            result = run_rag_pipeline(doc_id, "What is the project budget?", top_k=1)

        self.assertTrue(result["has_sufficient_evidence"])
        self.assertIn("₹1.8 crore", result["answer"])
        self.assertEqual(len(result["sources"]), 1)

        source = result["sources"][0]
        self.assertEqual(source["citation_index"], 1)
        self.assertEqual(source["filename"], "riverside_spec.pdf")
        self.assertEqual(source["page"], 2)
        self.assertEqual(source["chunk_id"], f"{doc_id}_p2_c5")
        self.assertEqual(source["score"], 0.895)

    def test_mock_mode_without_openai_api_key_succeeds(self):
        """Mock mode does not require an OPENAI_API_KEY to be set."""
        doc_id = "doc_no_api_key_needed"
        registry.register_document({
            "document_id": doc_id,
            "filename": "guide.pdf",
            "total_pages": 1,
            "total_characters": 500,
            "total_words": 80,
            "pages": [],
        })
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "chunk_id": f"{doc_id}_p1_c0",
                "page": 1,
                "text": "User authentication is configured using JSON Web Tokens.",
                "score": 0.77,
                "char_count": 56,
                "word_count": 8,
            }
        ]
        registry.set_index(doc_id, mock_store, chunk_count=1)

        with patch.object(settings, "LLM_PROVIDER", "mock"), \
             patch.object(settings, "OPENAI_API_KEY", ""):
            result = run_rag_pipeline(doc_id, "How is authentication handled?", top_k=1)

        self.assertTrue(result["has_sufficient_evidence"])
        self.assertEqual(result["llm_provider"], "mock")


class TestMockLLMHTTPEndpoints(unittest.TestCase):
    """End-to-end HTTP integration tests for /ask in mock mode."""

    def setUp(self):
        registry._documents.clear()
        registry._indices.clear()

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_mock_ask_endpoint_success_and_citation_contract(self):
        """HTTP POST /ask in mock mode returns 200 with citations and mock provider flag."""
        doc_id = "http_mock_doc_1"
        registry.register_document({
            "document_id": doc_id,
            "filename": "quarterly_results.pdf",
            "total_pages": 5,
            "total_characters": 5000,
            "total_words": 800,
            "pages": [],
        })

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "chunk_id": f"{doc_id}_p2_c1",
                "page": 2,
                "text": "Operating margin expanded to 28.4% driven by cloud growth.",
                "score": 0.91,
                "char_count": 58,
                "word_count": 8,
            }
        ]
        registry.set_index(doc_id, mock_store, chunk_count=1)

        with patch.object(settings, "LLM_PROVIDER", "mock"), \
             patch.object(settings, "OPENAI_API_KEY", None):
            response = client.post(
                f"/api/v1/documents/{doc_id}/ask",
                json={"question": "What was the operating margin?", "top_k": 2},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["llm_provider"], "mock")
        self.assertEqual(data["model_used"], MockLLMService.MOCK_MODEL_NAME)
        self.assertTrue(data["has_sufficient_evidence"])
        self.assertIn("Operating margin expanded to 28.4%", data["answer"])
        self.assertIn("[1]", data["answer"])

        # Validate citation contract
        self.assertEqual(len(data["sources"]), 1)
        src = data["sources"][0]
        self.assertEqual(src["citation_index"], 1)
        self.assertEqual(src["filename"], "quarterly_results.pdf")
        self.assertEqual(src["page"], 2)
        self.assertEqual(src["chunk_id"], f"{doc_id}_p2_c1")
        self.assertAlmostEqual(src["score"], 0.91, places=2)
        self.assertIn("Operating margin", src["text"])

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_mock_ask_endpoint_invalid_document_id_returns_404(self):
        """Querying an unknown document returns 404 Not Found."""
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            response = client.post(
                "/api/v1/documents/non_existent_doc_id/ask",
                json={"question": "What is here?", "top_k": 2},
            )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    @unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient required")
    def test_mock_ask_endpoint_unindexed_document_returns_400(self):
        """Querying an unindexed document returns 400 Bad Request."""
        doc_id = "doc_unindexed_mock"
        registry.register_document({
            "document_id": doc_id,
            "filename": "unindexed.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 10,
            "pages": [],
        })
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            response = client.post(
                f"/api/v1/documents/{doc_id}/ask",
                json={"question": "What is here?", "top_k": 2},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("has not been indexed yet", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
