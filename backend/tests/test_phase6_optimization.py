"""Phase 6.2 Performance, Caching and Resource Optimization Tests.

Validates:
1. QueryEmbeddingCache bounded capacity, LRU eviction, and hit/miss tracking.
2. EmbeddingService query caching and zero caching for oversized inputs.
3. FAISSVectorStore clear() explicitly releases index and chunks.
4. DocumentRegistry remove_document() and clear() invoke index cleanup.
5. DELETE /api/v1/documents/{document_id} releases session memory and index resources.
6. Repeated queries utilize cached embeddings without redundant encoding.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.embedding_service import QueryEmbeddingCache, EmbeddingService, embedding_service
from app.services.vector_store import FAISSVectorStore
from app.services.chunking_service import TextChunk
from app.services.document_registry import DocumentRegistry, registry


class TestPhase6Optimization(unittest.TestCase):
    """Tests for Phase 6.2 performance, caching, and resource lifecycle."""

    def setUp(self):
        self.client = TestClient(app)
        registry.clear()
        embedding_service.clear_cache()
        settings.LLM_PROVIDER = "mock"

    def tearDown(self):
        registry.clear()
        embedding_service.clear_cache()

    def test_query_cache_hit_miss_and_lru_eviction(self):
        """QueryEmbeddingCache must track hits/misses and enforce strict LRU eviction."""
        cache = QueryEmbeddingCache(max_size=3)
        v1 = np.ones((1, 384), dtype=np.float32)
        v2 = np.full((1, 384), 2.0, dtype=np.float32)
        v3 = np.full((1, 384), 3.0, dtype=np.float32)
        v4 = np.full((1, 384), 4.0, dtype=np.float32)

        # 1. Misses
        self.assertIsNone(cache.get("query 1"))
        self.assertEqual(cache.misses, 1)

        # Insert 3 entries
        cache.put("query 1", v1)
        cache.put("query 2", v2)
        cache.put("query 3", v3)
        self.assertEqual(cache.stats()["size"], 3)
        self.assertEqual(cache.stats()["evictions"], 0)

        # 2. Access "query 1" -> becomes MRU (hit)
        res1 = cache.get("query 1")
        self.assertIsNotNone(res1)
        np.testing.assert_array_equal(res1, v1)
        self.assertEqual(cache.hits, 1)

        # 3. Insert 4th entry -> "query 2" was LRU, so "query 2" must be evicted
        cache.put("query 4", v4)
        self.assertEqual(cache.stats()["size"], 3)
        self.assertEqual(cache.stats()["evictions"], 1)

        self.assertIsNone(cache.get("query 2"), "query 2 should have been evicted")
        self.assertIsNotNone(cache.get("query 1"), "query 1 should remain in cache")
        self.assertIsNotNone(cache.get("query 3"))
        self.assertIsNotNone(cache.get("query 4"))

    def test_query_cache_ignores_oversized_queries(self):
        """QueryEmbeddingCache must not store excessively long queries (>500 chars)."""
        cache = QueryEmbeddingCache(max_size=10)
        long_query = "a" * 550
        dummy_vec = np.zeros((1, 384), dtype=np.float32)

        cache.put(long_query, dummy_vec)
        self.assertEqual(cache.stats()["size"], 0)
        self.assertIsNone(cache.get(long_query))

    def test_embedding_service_embed_query_uses_cache(self):
        """embed_query must return cached vectors on repeat calls without re-encoding."""
        svc = EmbeddingService()
        mock_model = MagicMock()
        mock_vec = np.ones((1, 384), dtype=np.float32)
        mock_model.encode.return_value = mock_vec
        svc.set_mock_model(mock_model)

        q = "What is the total budget allocated?"

        # Call 1: Cache miss -> model.encode invoked
        res1 = svc.embed_query(q)
        self.assertEqual(mock_model.encode.call_count, 1)
        self.assertEqual(svc.query_cache.hits, 0)
        self.assertEqual(svc.query_cache.misses, 1)

        # Call 2: Cache hit -> model.encode NOT invoked again
        res2 = svc.embed_query(q)
        self.assertEqual(mock_model.encode.call_count, 1, "Model encode must not be called on cache hit")
        self.assertEqual(svc.query_cache.hits, 1)
        np.testing.assert_array_equal(res1, res2)

    def test_faiss_vector_store_clear_releases_resources(self):
        """FAISSVectorStore.clear() must reset index and clear chunk lists."""
        vs = FAISSVectorStore(dimension=384)
        chunks = [
            TextChunk(
                chunk_id="chk_1",
                document_id="doc_res",
                source_filename="test.pdf",
                page=1,
                chunk_index=0,
                text="Some test passage.",
                char_count=18,
                word_count=3,
            )
        ]
        embeddings = np.ones((1, 384), dtype=np.float32)

        vs.build_index(chunks, embeddings)
        self.assertEqual(len(vs.chunks), 1)
        self.assertIsNotNone(vs.index)

        # Invoke clear
        vs.clear()
        self.assertEqual(len(vs.chunks), 0)
        self.assertIsNone(vs.index)

    def test_document_registry_remove_and_clear_releases_vector_store(self):
        """DocumentRegistry removal must trigger clear() on the associated vector store."""
        reg = DocumentRegistry(max_capacity=5)
        mock_vs = MagicMock()

        reg.register_document({
            "document_id": "doc_cleanup_1",
            "filename": "cleanup.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 15,
            "pages": [{"page": 1, "text": "Content"}],
        })
        reg.set_index("doc_cleanup_1", mock_vs, 5)

        # Remove single document
        reg.remove_document("doc_cleanup_1")
        mock_vs.clear.assert_called_once()
        self.assertIsNone(reg.get_document("doc_cleanup_1"))
        self.assertIsNone(reg.get_index("doc_cleanup_1"))

        # Test reg.clear() also cleans up all remaining indices
        mock_vs_2 = MagicMock()
        reg.register_document({
            "document_id": "doc_cleanup_2",
            "filename": "cleanup2.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 15,
            "pages": [{"page": 1, "text": "Content"}],
        })
        reg.set_index("doc_cleanup_2", mock_vs_2, 3)

        reg.clear()
        mock_vs_2.clear.assert_called_once()
        self.assertEqual(len(reg._documents), 0)
        self.assertEqual(len(reg._indices), 0)

    def test_delete_endpoint_clears_document_and_vector_store(self):
        """DELETE /api/v1/documents/{document_id} cleans up document from memory and releases FAISS index."""
        vs = FAISSVectorStore(dimension=384)
        chunks = [
            TextChunk(
                chunk_id="chk_del_1",
                document_id="doc_del_api",
                source_filename="sample.pdf",
                page=1,
                chunk_index=0,
                text="Text to be indexed.",
                char_count=19,
                word_count=4,
            )

        ]
        embeddings = np.ones((1, 384), dtype=np.float32)
        vs.build_index(chunks, embeddings)

        registry.register_document({
            "document_id": "doc_del_api",
            "filename": "sample.pdf",
            "total_pages": 1,
            "total_characters": 100,
            "total_words": 20,
            "pages": [{"page": 1, "text": "Text to be indexed."}],
        })
        registry.set_index("doc_del_api", vs, 1)

        self.assertTrue(registry.is_indexed("doc_del_api"))
        self.assertEqual(len(vs.chunks), 1)

        # Call DELETE endpoint
        res = self.client.delete("/api/v1/documents/doc_del_api")
        self.assertEqual(res.status_code, 200)

        # Confirm registry is empty and FAISS store was cleared
        self.assertIsNone(registry.get_document("doc_del_api"))
        self.assertIsNone(registry.get_index("doc_del_api"))
        self.assertEqual(len(vs.chunks), 0)
        self.assertIsNone(vs.index)


if __name__ == "__main__":
    unittest.main()
