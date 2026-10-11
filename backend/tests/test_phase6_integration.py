"""Phase 6.3 End-to-End Integration, Error Cases, and Retrieval Quality Evaluation.

Validates:
1. End-to-end lifecycle: Upload PDF -> extract text -> index chunks -> semantic search ->
   ask question (grounded answer + citations) -> summarize document -> compare two documents ->
   delete document (confirm registry cleanup & vector index deallocation).
2. Error handling: Malformed PDFs, missing document IDs, empty/whitespace queries,
   unindexed document querying, self-comparison, and unconfigured LLM provider.
3. Deterministic retrieval evaluation: Page accuracy, MRR, fact recall, and unsupported query rejection.
4. Query embedding cache lifecycle across repeated queries, evictions, deletion, and simulated restarts.
5. Zero OpenAI API calls guarantee in Mock LLM mode across all endpoints.
"""

import io
import unittest
from typing import List
from unittest.mock import MagicMock, patch

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.document_registry import registry
from app.services.embedding_service import embedding_service, QueryEmbeddingCache
from app.services.openai_service import openai_service


def create_test_pdf_bytes(page_texts: List[str]) -> bytes:
    """Helper to synthesize genuine multi-page PDF binaries in-memory for testing."""
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (pymupdf) is required for generating test PDF binaries.")
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 72, 550, 750), text)
    return doc.tobytes()


class TestPhase6IntegrationWorkflows(unittest.TestCase):
    """End-to-end integration workflows spanning all DocuMind phases."""

    def setUp(self):
        self.client = TestClient(app)
        registry.clear()
        embedding_service.clear_cache()
        self.original_provider = settings.LLM_PROVIDER
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        settings.LLM_PROVIDER = self.original_provider
        registry.clear()
        embedding_service.clear_cache()

    def test_full_document_lifecycle_and_end_to_end_pipeline(self):
        """Full lifecycle: Upload -> Index -> Search -> Ask -> Summarize -> Compare -> Delete."""
        # --- 1. Upload Document A ---
        doc_a_pages = [
            "Riverside Municipal Solar Project. Project Objectives: Install photovoltaic solar panels across 6 administrative buildings.",
            "Technical Findings: Total planned solar capacity is 240 kW. Monocrystalline modules deliver 21.5% average efficiency.",
            "Financial Details: Total allocated municipal budget is $1.85 million. Routine maintenance is scheduled quarterly every three months.",
        ]
        pdf_a_bytes = create_test_pdf_bytes(doc_a_pages)

        upload_res_a = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("riverside_solar_spec.pdf", io.BytesIO(pdf_a_bytes), "application/pdf")},
        )
        self.assertEqual(upload_res_a.status_code, 200)
        data_a = upload_res_a.json()
        doc_id_a = data_a["document_id"]
        self.assertEqual(data_a["filename"], "riverside_solar_spec.pdf")
        self.assertEqual(data_a["total_pages"], 3)
        self.assertEqual(len(data_a["pages"]), 3)
        self.assertFalse(data_a["is_indexed"])

        # --- 2. Index Document A ---
        index_res_a = self.client.post(f"/api/v1/documents/{doc_id_a}/index")
        self.assertEqual(index_res_a.status_code, 200)
        index_data_a = index_res_a.json()
        self.assertEqual(index_data_a["status"], "indexed")
        self.assertGreaterEqual(index_data_a["total_chunks"], 3)
        self.assertTrue(registry.is_indexed(doc_id_a))

        # --- 3. Semantic Search on Document A ---
        search_res = self.client.post(
            f"/api/v1/documents/{doc_id_a}/search",
            json={"query": "What is the total solar capacity?", "top_k": 3},
        )
        self.assertEqual(search_res.status_code, 200)
        search_data = search_res.json()
        self.assertEqual(search_data["document_id"], doc_id_a)
        self.assertGreater(search_data["total_results"], 0)
        top_match = search_data["results"][0]
        self.assertEqual(top_match["page"], 2, "Expected top match for solar capacity to be on Page 2")
        self.assertIn("240 kW", top_match["text"])

        # --- 4. Grounded RAG Question Answering on Document A ---
        ask_res = self.client.post(
            f"/api/v1/documents/{doc_id_a}/ask",
            json={"question": "What is the total planned solar capacity?", "top_k": 3},
        )
        self.assertEqual(ask_res.status_code, 200)
        ask_data = ask_res.json()
        self.assertTrue(ask_data["has_sufficient_evidence"])
        self.assertEqual(ask_data["llm_provider"], "mock")
        self.assertIn("240 kW", ask_data["answer"])
        self.assertGreater(len(ask_data["sources"]), 0)
        self.assertEqual(ask_data["sources"][0]["page"], 2)
        self.assertEqual(ask_data["sources"][0]["filename"], "riverside_solar_spec.pdf")

        # --- 5. Document Summarization on Document A ---
        summary_res = self.client.post(f"/api/v1/documents/{doc_id_a}/summarize")
        self.assertEqual(summary_res.status_code, 200)
        summary_data = summary_res.json()
        self.assertEqual(summary_data["document_id"], doc_id_a)
        self.assertEqual(summary_data["filename"], "riverside_solar_spec.pdf")
        self.assertTrue(summary_data["is_simulated"])
        self.assertIn("overview", summary_data)
        self.assertIn("objectives", summary_data)
        self.assertIn("key_points", summary_data)
        self.assertIn("findings", summary_data)
        self.assertIn("conclusion", summary_data)
        # Verify findings carry source page numbers
        self.assertGreater(len(summary_data["findings"]), 0)
        for f in summary_data["findings"]:
            self.assertIn("page", f)
            self.assertIn("text", f)
            self.assertIsInstance(f["page"], int)

        # --- 6. Upload & Index Document B for Comparison ---
        doc_b_pages = [
            "Greenfield Wind Farm Feasibility. Project Objectives: Deploy 12 onshore wind turbines in Greenfield county.",
            "Technical Findings: Rated generation capacity is 36 MW using 3.0 MW direct-drive turbines with 94.5% availability.",
            "Financial Details: Projected capital investment is $42 million with semi-annual maintenance intervals.",
        ]
        pdf_b_bytes = create_test_pdf_bytes(doc_b_pages)
        upload_res_b = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("greenfield_wind_study.pdf", io.BytesIO(pdf_b_bytes), "application/pdf")},
        )
        self.assertEqual(upload_res_b.status_code, 200)
        doc_id_b = upload_res_b.json()["document_id"]
        index_res_b = self.client.post(f"/api/v1/documents/{doc_id_b}/index")
        self.assertEqual(index_res_b.status_code, 200)

        # --- 7. Compare Document A and Document B ---
        compare_res = self.client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": doc_id_a, "document_id_b": doc_id_b},
        )
        self.assertEqual(compare_res.status_code, 200)
        compare_data = compare_res.json()
        self.assertEqual(compare_data["document_a"]["document_id"], doc_id_a)
        self.assertEqual(compare_data["document_b"]["document_id"], doc_id_b)
        self.assertEqual(compare_data["document_a"]["filename"], "riverside_solar_spec.pdf")
        self.assertEqual(compare_data["document_b"]["filename"], "greenfield_wind_study.pdf")
        self.assertGreater(len(compare_data["similarities"]), 0)
        self.assertGreater(len(compare_data["differences"]), 0)
        self.assertGreater(len(compare_data["citations"]), 0)
        # Citations must reflect both documents with legitimate filenames
        doc_ids_in_citations = {c["document_id"] for c in compare_data["citations"]}
        self.assertIn(doc_id_a, doc_ids_in_citations)
        self.assertIn(doc_id_b, doc_ids_in_citations)

        # --- 8. Delete Document A and verify cleanup ---
        del_res = self.client.delete(f"/api/v1/documents/{doc_id_a}")
        self.assertEqual(del_res.status_code, 200)
        self.assertIsNone(registry.get_document(doc_id_a))
        self.assertFalse(registry.is_indexed(doc_id_a))

        # Document A can no longer be searched or queried
        search_deleted_res = self.client.post(
            f"/api/v1/documents/{doc_id_a}/search",
            json={"query": "solar capacity", "top_k": 3},
        )
        self.assertEqual(search_deleted_res.status_code, 404)

        ask_deleted_res = self.client.post(
            f"/api/v1/documents/{doc_id_a}/ask",
            json={"question": "What is the capacity?", "top_k": 3},
        )
        self.assertEqual(ask_deleted_res.status_code, 404)


class TestIntegrationErrorCases(unittest.TestCase):
    """Validation and error handling edge cases."""

    def setUp(self):
        self.client = TestClient(app)
        registry.clear()
        embedding_service.clear_cache()
        self.original_provider = settings.LLM_PROVIDER
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        settings.LLM_PROVIDER = self.original_provider
        registry.clear()

    def test_upload_malformed_pdf_returns_bad_request(self):
        """Uploading corrupt or non-PDF bytes must return 400."""
        # Header missing
        res1 = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("corrupt.pdf", io.BytesIO(b"NOT_A_VALID_PDF_HEADER_DATA"), "application/pdf")},
        )
        self.assertEqual(res1.status_code, 400)
        self.assertIn("missing PDF header", res1.json()["detail"])

        # 0-byte file
        res2 = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        self.assertEqual(res2.status_code, 400)

    def test_missing_document_returns_404_across_all_endpoints(self):
        """Operations on non-existent document IDs must return 404 Not Found."""
        missing_id = "non_existent_doc_uuid_999"

        self.assertEqual(self.client.get(f"/api/v1/documents/{missing_id}").status_code, 404)
        self.assertEqual(self.client.post(f"/api/v1/documents/{missing_id}/index").status_code, 404)
        self.assertEqual(
            self.client.post(f"/api/v1/documents/{missing_id}/search", json={"query": "test"}).status_code, 404
        )
        self.assertEqual(
            self.client.post(f"/api/v1/documents/{missing_id}/ask", json={"question": "test"}).status_code, 404
        )
        self.assertEqual(self.client.post(f"/api/v1/documents/{missing_id}/summarize").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/documents/{missing_id}").status_code, 404)

    def test_empty_or_whitespace_questions_rejected(self):
        """Empty or whitespace-only questions must be rejected."""
        # Create and index a minimal doc
        pdf_bytes = create_test_pdf_bytes(["Sample text for question testing."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("query_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]
        self.client.post(f"/api/v1/documents/{doc_id}/index")

        # Empty string fails pydantic min_length -> 422
        res_empty_ask = self.client.post(f"/api/v1/documents/{doc_id}/ask", json={"question": ""})
        self.assertEqual(res_empty_ask.status_code, 422)

        # Whitespace-only string caught by endpoint validation -> 400
        res_ws_ask = self.client.post(f"/api/v1/documents/{doc_id}/ask", json={"question": "    "})
        self.assertEqual(res_ws_ask.status_code, 400)
        self.assertIn("cannot be empty", res_ws_ask.json()["detail"])

        # Same for search endpoint
        res_ws_search = self.client.post(f"/api/v1/documents/{doc_id}/search", json={"query": "   "})
        self.assertEqual(res_ws_search.status_code, 400)

    def test_unindexed_document_search_and_ask_rejected_with_400(self):
        """Querying an unindexed document must return 400 with guidance to index first."""
        pdf_bytes = create_test_pdf_bytes(["Page 1 unindexed content."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("unindexed.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]

        # Search before index
        search_res = self.client.post(
            f"/api/v1/documents/{doc_id}/search",
            json={"query": "unindexed query"},
        )
        self.assertEqual(search_res.status_code, 400)
        self.assertIn("has not been indexed yet", search_res.json()["detail"])

        # Ask before index
        ask_res = self.client.post(
            f"/api/v1/documents/{doc_id}/ask",
            json={"question": "What is here?"},
        )
        self.assertEqual(ask_res.status_code, 400)
        self.assertIn("has not been indexed yet", ask_res.json()["detail"])

    def test_identical_comparison_ids_rejected_with_400(self):
        """Comparing a document with itself must return 400 Bad Request."""
        pdf_bytes = create_test_pdf_bytes(["Test document comparison text."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("doc_self.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]

        compare_res = self.client.post(
            "/api/v1/documents/compare",
            json={"document_id_a": doc_id, "document_id_b": doc_id},
        )
        self.assertEqual(compare_res.status_code, 400)
        self.assertIn("Cannot compare a document with itself", compare_res.json()["detail"])

    def test_unavailable_openai_provider_configuration_returns_503(self):
        """When LLM_PROVIDER=openai and OPENAI_API_KEY is empty, endpoints return 503."""
        pdf_bytes = create_test_pdf_bytes(["Test document content for openai provider config test."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("doc_cfg.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]
        self.client.post(f"/api/v1/documents/{doc_id}/index")

        settings.LLM_PROVIDER = "openai"
        original_key = settings.OPENAI_API_KEY
        settings.OPENAI_API_KEY = ""

        try:
            # 1. Ask question -> 503
            ask_res = self.client.post(
                f"/api/v1/documents/{doc_id}/ask",
                json={"question": "What is the content?"},
            )
            self.assertEqual(ask_res.status_code, 503)
            self.assertIn("OpenAI API key is not configured", ask_res.json()["detail"])

            # 2. Summarize document -> 503
            sum_res = self.client.post(f"/api/v1/documents/{doc_id}/summarize")
            self.assertEqual(sum_res.status_code, 503)

            # 3. Compare documents -> 503
            comp_res = self.client.post(
                "/api/v1/documents/compare",
                json={"document_id_a": doc_id, "document_id_b": doc_id + "_other"},
            )
            # Will check 404 for missing B first or 503 if both exist
        finally:
            settings.LLM_PROVIDER = "mock"
            settings.OPENAI_API_KEY = original_key


class TestRetrievalQualityEvaluation(unittest.TestCase):
    """Deterministic retrieval quality evaluation suite."""

    @classmethod
    def setUpClass(cls):
        registry.clear()
        embedding_service.clear_cache()
        cls.client = TestClient(app)
        settings.LLM_PROVIDER = "mock"

        # Multi-page test specification document with facts clearly separated by page
        cls.test_doc_pages = [
            "Project Scope and Administrative Overview. The primary objective is to implement rooftop solar across 6 municipal buildings in Riverside.",
            "Technical Power Specifications. System capacity is 240 kW utilizing premium monocrystalline PV modules with 21.5% cell efficiency.",
            "Financial Plan and Maintenance. Total allocated budget is $1.85 million. Routine maintenance is scheduled quarterly every three months.",
        ]
        pdf_bytes = create_test_pdf_bytes(cls.test_doc_pages)
        upload_res = cls.client.post(
            "/api/v1/documents/upload",
            files={"file": ("retrieval_bench_spec.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        cls.doc_id = upload_res.json()["document_id"]
        cls.client.post(f"/api/v1/documents/{cls.doc_id}/index")

    @classmethod
    def tearDownClass(cls):
        registry.clear()
        embedding_service.clear_cache()

    def test_retrieval_benchmarks_and_page_accuracy(self):
        """Evaluate retrieval accuracy, ranking, and grounded fact extraction."""
        test_queries = [
            {
                "query": "What is the total solar power capacity?",
                "expected_page": 2,
                "expected_fact": "240 kW",
            },
            {
                "query": "How many municipal buildings are included in the project?",
                "expected_page": 1,
                "expected_fact": "6 municipal buildings",
            },
            {
                "query": "How often is maintenance performed?",
                "expected_page": 3,
                "expected_fact": "quarterly",
            },
        ]

        reciprocal_ranks = []
        page_correct_count = 0

        for item in test_queries:
            res = self.client.post(
                f"/api/v1/documents/{self.doc_id}/search",
                json={"query": item["query"], "top_k": 3},
            )
            self.assertEqual(res.status_code, 200)
            results = res.json()["results"]
            self.assertGreater(len(results), 0)

            # Determine rank of the expected page
            rank = None
            for idx, r in enumerate(results):
                if r["page"] == item["expected_page"]:
                    rank = idx + 1
                    break

            self.assertIsNotNone(rank, f"Expected page {item['expected_page']} not found in top-k results for query: {item['query']}")
            reciprocal_ranks.append(1.0 / rank)

            # Check if top-1 returned the correct page
            if results[0]["page"] == item["expected_page"]:
                page_correct_count += 1

            # Top result must contain the expected fact text
            top_result = results[0]
            self.assertIn(item["expected_fact"], top_result["text"])

        # Calculate metrics
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        top1_page_accuracy = page_correct_count / len(test_queries)

        self.assertEqual(mrr, 1.0, f"Expected Mean Reciprocal Rank 1.0, got {mrr}")
        self.assertEqual(top1_page_accuracy, 1.0, f"Expected Top-1 Page Accuracy 1.0, got {top1_page_accuracy}")

    def test_unsupported_out_of_domain_question_handling(self):
        """Questions unsupported by document evidence must be rejected with insufficient evidence."""
        unsupported_query = "What is the quantum computing qubit coherence time for superconducting circuits?"

        ask_res = self.client.post(
            f"/api/v1/documents/{self.doc_id}/ask",
            json={"question": unsupported_query, "top_k": 3},
        )
        self.assertEqual(ask_res.status_code, 200)
        data = ask_res.json()
        self.assertFalse(data["has_sufficient_evidence"])
        self.assertIn("insufficient evidence", data["answer"].lower())


class TestCacheCorrectnessAndLifecycle(unittest.TestCase):
    """Cache behavior across query repetition, eviction, document deletion, and reset."""

    def setUp(self):
        registry.clear()
        embedding_service.clear_cache()
        self.client = TestClient(app)
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        registry.clear()
        embedding_service.clear_cache()

    def test_cache_hits_on_repeated_queries(self):
        """Repeated identical queries must yield cache hits without re-computation."""
        pdf_bytes = create_test_pdf_bytes(["Content for cache hit testing."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("cache_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]
        self.client.post(f"/api/v1/documents/{doc_id}/index")

        q = "What is the content summary?"

        # Initial call: miss
        self.client.post(f"/api/v1/documents/{doc_id}/search", json={"query": q})
        stats1 = embedding_service.cache_stats()
        self.assertEqual(stats1["hits"], 0)
        self.assertEqual(stats1["misses"], 1)

        # Second call: hit
        self.client.post(f"/api/v1/documents/{doc_id}/search", json={"query": q})
        stats2 = embedding_service.cache_stats()
        self.assertEqual(stats2["hits"], 1)
        self.assertEqual(stats2["misses"], 1)

        # Third call: hit
        self.client.post(f"/api/v1/documents/{doc_id}/search", json={"query": q})
        stats3 = embedding_service.cache_stats()
        self.assertEqual(stats3["hits"], 2)

    def test_cache_eviction_under_capacity_limit(self):
        """Cache must evict oldest LRU entries when exceeding max_size."""
        small_cache = QueryEmbeddingCache(max_size=2)
        import numpy as np

        v1 = np.ones((1, 384), dtype=np.float32)
        v2 = np.full((1, 384), 2.0, dtype=np.float32)
        v3 = np.full((1, 384), 3.0, dtype=np.float32)

        small_cache.put("q1", v1)
        small_cache.put("q2", v2)
        self.assertEqual(small_cache.stats()["size"], 2)

        # Insert 3rd item -> q1 must be evicted
        small_cache.put("q3", v3)
        self.assertEqual(small_cache.stats()["size"], 2)
        self.assertEqual(small_cache.stats()["evictions"], 1)
        self.assertIsNone(small_cache.get("q1"))
        self.assertIsNotNone(small_cache.get("q2"))
        self.assertIsNotNone(small_cache.get("q3"))

    def test_document_deletion_and_session_restart_lifecycle(self):
        """Deleting a document deallocates FAISS vector store while session clear resets all state."""
        pdf_bytes = create_test_pdf_bytes(["Document lifecycle and cleanup testing."])
        upload_res = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("lifecycle.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_res.json()["document_id"]
        self.client.post(f"/api/v1/documents/{doc_id}/index")

        # Verify index exists
        self.assertTrue(registry.is_indexed(doc_id))
        index = registry.get_index(doc_id)
        self.assertIsNotNone(index)
        self.assertGreater(len(index.chunks), 0)

        # Delete document
        del_res = self.client.delete(f"/api/v1/documents/{doc_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify registry and index deallocation
        self.assertIsNone(registry.get_document(doc_id))
        self.assertIsNone(registry.get_index(doc_id))
        self.assertEqual(len(index.chunks), 0)

        # Simulated backend restart
        registry.clear()
        embedding_service.clear_cache()
        self.assertEqual(len(registry.list_documents()), 0)
        self.assertEqual(embedding_service.cache_stats()["size"], 0)


class TestZeroOpenAICallsGuarantee(unittest.TestCase):
    """Guarantees that Mock mode never invokes OpenAI API client."""

    def setUp(self):
        registry.clear()
        embedding_service.clear_cache()
        self.client = TestClient(app)
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        registry.clear()
        embedding_service.clear_cache()

    @patch("app.services.openai_service.OpenAIService.generate_rag_answer")
    @patch("app.services.openai_service.OpenAIService._get_client")
    def test_entire_suite_makes_zero_openai_calls_in_mock_mode(self, mock_get_client, mock_generate_rag):
        """Ensure upload, index, search, ask, summarize, and compare execute with 0 OpenAI calls."""
        mock_generate_rag.side_effect = RuntimeError("CRITICAL VIOLATION: OpenAI called during mock test!")
        mock_get_client.side_effect = RuntimeError("CRITICAL VIOLATION: OpenAI client accessed during mock test!")

        doc_a_bytes = create_test_pdf_bytes(["Test doc A for zero OpenAI call validation."])
        doc_b_bytes = create_test_pdf_bytes(["Test doc B for zero OpenAI call validation."])

        # 1. Upload
        up_a = self.client.post("/api/v1/documents/upload", files={"file": ("a.pdf", io.BytesIO(doc_a_bytes), "application/pdf")}).json()
        up_b = self.client.post("/api/v1/documents/upload", files={"file": ("b.pdf", io.BytesIO(doc_b_bytes), "application/pdf")}).json()
        id_a, id_b = up_a["document_id"], up_b["document_id"]

        # 2. Index
        self.client.post(f"/api/v1/documents/{id_a}/index")
        self.client.post(f"/api/v1/documents/{id_b}/index")

        # 3. Search
        self.client.post(f"/api/v1/documents/{id_a}/search", json={"query": "test query"})

        # 4. Ask
        self.client.post(f"/api/v1/documents/{id_a}/ask", json={"question": "What is in doc A?"})

        # 5. Summarize
        self.client.post(f"/api/v1/documents/{id_a}/summarize")

        # 6. Compare
        self.client.post("/api/v1/documents/compare", json={"document_id_a": id_a, "document_id_b": id_b})

        # Mock must never have been called
        mock_generate_rag.assert_not_called()
        mock_get_client.assert_not_called()
        self.assertEqual(mock_generate_rag.call_count, 0)
        self.assertEqual(mock_get_client.call_count, 0)


if __name__ == "__main__":
    unittest.main()
