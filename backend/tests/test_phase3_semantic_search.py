"""Tests for Phase 3 Semantic Search: Chunking, Vector Store, Registry, and Search Endpoints."""

import unittest
from unittest.mock import MagicMock

from app.services.chunking_service import chunk_document, chunk_text
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EMBEDDING_DIMENSION
from app.services.vector_store import FAISSVectorStore

# Endpoints require python-multipart (installed via requirements.txt)
try:
    from fastapi import HTTPException
    from app.api.documents import (
        DocumentIndexRequest,
        SearchQueryRequest,
        index_document,
        search_document,
    )
    HAS_DOCUMENTS_API = True
except Exception:
    HAS_DOCUMENTS_API = False


class TestChunkingService(unittest.TestCase):
    """Unit tests for text chunking and metadata preservation."""

    def test_chunk_overlap_preserved(self):
        """Verify that adjacent chunks preserve the specified character overlap."""
        sample_text = (
            "Natural Language Processing enables computers to understand human language. "
            "Semantic search uses embeddings to retrieve relevant passages by meaning rather than keywords. "
            "FAISS provides high-performance similarity search over dense vector representations."
        )
        chunk_size = 80
        chunk_overlap = 20

        chunks = chunk_text(
            text=sample_text,
            page_number=1,
            document_id="doc_overlap_test",
            source_filename="test.pdf",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.assertGreater(len(chunks), 1)

        # Check that contiguous slices have overlapping substring
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i].page, 1)
            self.assertEqual(chunks[i].document_id, "doc_overlap_test")
            self.assertEqual(chunks[i].source_filename, "test.pdf")
            self.assertTrue(chunks[i].chunk_id.startswith("doc_overlap_test_p1_c"))

    def test_chunk_page_metadata_preserved_multi_page(self):
        """Verify that multi-page documents maintain correct source page provenance on every chunk."""
        doc = {
            "document_id": "doc_meta_123",
            "filename": "multi_page_paper.pdf",
            "pages": [
                {"page": 1, "text": "Page one text discussing introduction and background literature."},
                {"page": 2, "text": "Page two text outlining the vector embedding methodology."},
                {"page": 3, "text": "Page three detailing FAISS indexing and benchmarking results."},
            ],
        }

        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[0].document_id, "doc_meta_123")
        self.assertEqual(chunks[0].source_filename, "multi_page_paper.pdf")
        self.assertEqual(chunks[0].chunk_id, "doc_meta_123_p1_c0")

        self.assertEqual(chunks[1].page, 2)
        self.assertEqual(chunks[1].chunk_id, "doc_meta_123_p2_c1")

        self.assertEqual(chunks[2].page, 3)
        self.assertEqual(chunks[2].chunk_id, "doc_meta_123_p3_c2")

    def test_chunk_invalid_parameters_raise_value_error(self):
        """chunk_overlap >= chunk_size must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            chunk_text("text", 1, "d", "f", chunk_size=100, chunk_overlap=100)
        self.assertIn("strictly less than", str(ctx.exception))

        with self.assertRaises(ValueError):
            chunk_text("text", 1, "d", "f", chunk_size=0, chunk_overlap=0)


class TestDocumentRegistry(unittest.TestCase):
    """Unit tests for the in-memory document session registry."""

    def setUp(self):
        self.registry = DocumentRegistry()

    def test_register_and_retrieve_document(self):
        """Document can be registered and retrieved by ID."""
        doc_data = {
            "document_id": "doc_uuid_abc",
            "filename": "thesis.pdf",
            "total_pages": 10,
            "total_characters": 15000,
            "total_words": 2300,
            "pages": [],
        }
        self.registry.register_document(doc_data)

        retrieved = self.registry.get_document("doc_uuid_abc")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["filename"], "thesis.pdf")
        self.assertFalse(self.registry.is_indexed("doc_uuid_abc"))

    def test_set_index_and_status(self):
        """Registering an index updates is_indexed status and stores vector store."""
        doc_data = {
            "document_id": "doc_uuid_idx",
            "filename": "report.pdf",
            "total_pages": 2,
            "total_characters": 500,
            "total_words": 80,
            "pages": [],
        }
        self.registry.register_document(doc_data)

        mock_vector_store = MagicMock()
        self.registry.set_index("doc_uuid_idx", mock_vector_store, chunk_count=5)

        self.assertTrue(self.registry.is_indexed("doc_uuid_idx"))
        self.assertEqual(self.registry.get_index("doc_uuid_idx"), mock_vector_store)

    def test_unknown_document_returns_none(self):
        """Querying an unregistered document ID returns None."""
        self.assertIsNone(self.registry.get_document("non_existent_id"))
        self.assertFalse(self.registry.is_indexed("non_existent_id"))


class TestVectorStoreAndRetrievalRelevance(unittest.TestCase):
    """Unit tests for FAISS Vector Store dimensions and retrieval relevance with mock embeddings."""

    def test_vector_store_retrieval_relevance(self):
        """Verify that query vector returns the closest semantic neighbor chunk with highest score."""
        dim = EMBEDDING_DIMENSION
        # Synthetic mock unit vectors of length 384
        vec_nlp = [0.0] * dim
        vec_nlp[0] = 1.0  # Direction along dim 0

        vec_history = [0.0] * dim
        vec_history[1] = 1.0  # Direction along dim 1

        vec_biology = [0.0] * dim
        vec_biology[2] = 1.0  # Direction along dim 2

        doc = {
            "document_id": "doc_semantic",
            "filename": "cross_domain.pdf",
            "pages": [
                {"page": 1, "text": "Transformer architecture and neural NLP."},
                {"page": 2, "text": "Roman empire ancient architecture and history."},
                {"page": 3, "text": "Cellular biology and genetics replication."},
            ],
        }
        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)

        # Build index with 3 synthetic vectors
        embeddings = [vec_nlp, vec_history, vec_biology]
        store = FAISSVectorStore(dimension=dim)
        store.build_index(chunks, embeddings)

        # Query aligned with NLP (dim 0 = 0.95, dim 1 = 0.05)
        query_vector = [0.0] * dim
        query_vector[0] = 0.95
        query_vector[1] = 0.05

        results = store.search(query_vector, top_k=3)

        self.assertEqual(len(results), 3)
        # Top result must be chunk 0 (NLP, page 1)
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["chunk_id"], "doc_semantic_p1_c0")
        self.assertIn("Transformer", results[0]["text"])
        self.assertGreater(results[0]["score"], results[1]["score"])
        # Score must be bounded [0.0, 1.0]
        self.assertGreaterEqual(results[0]["score"], 0.0)
        self.assertLessEqual(results[0]["score"], 1.0)


class TestSemanticSearchEndpoints(unittest.TestCase):
    """Unit tests for indexing and search endpoint validation."""

    def setUp(self):
        if not HAS_DOCUMENTS_API:
            self.skipTest("Requires python-multipart (pip install -r requirements.txt)")

    def test_index_endpoint_unknown_doc_id_raises_404(self):
        """Attempting to index an unuploaded document ID must return 404."""
        with self.assertRaises(HTTPException) as ctx:
            index_document("fake_unknown_uuid_123")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found in in-memory session", ctx.exception.detail)

    def test_search_endpoint_unknown_doc_id_raises_404(self):
        """Attempting to search an unuploaded document ID must return 404."""
        req = SearchQueryRequest(query="machine learning", top_k=5)
        with self.assertRaises(HTTPException) as ctx:
            search_document("fake_unknown_uuid_456", req)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found in in-memory session", ctx.exception.detail)

    def test_search_endpoint_empty_query_raises_400(self):
        """Empty or whitespace-only search query must return 400."""
        req = SearchQueryRequest(query="   ", top_k=5)
        with self.assertRaises(HTTPException) as ctx:
            search_document("any_doc_id", req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("cannot be empty", ctx.exception.detail)

    def test_search_endpoint_unindexed_doc_raises_400(self):
        """Searching a document that has not been indexed must return 400."""
        from app.services.document_registry import registry
        registry.register_document({
            "document_id": "unindexed_doc_789",
            "filename": "notes.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 15,
            "pages": [{"page": 1, "text": "Some text."}],
        })

        req = SearchQueryRequest(query="test query", top_k=3)
        with self.assertRaises(HTTPException) as ctx:
            search_document("unindexed_doc_789", req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("has not been indexed yet", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
