"""FAISS vector store service for similarity search using inner-product (cosine similarity)."""

from typing import Any, Dict, List, Optional
from app.services.chunking_service import TextChunk

DEFAULT_DIMENSION = 384


class FAISSVectorStore:
    """Manages an in-memory FAISS IndexFlatIP index for a single document."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self.chunks: List[TextChunk] = []
        self.index: Optional[Any] = None

    def clear(self) -> None:
        """Explicitly release in-memory index, chunk lists, and fallback embedding arrays."""
        if hasattr(self, "index") and self.index is not None and self.index != "fallback":
            try:
                if hasattr(self.index, "reset"):
                    self.index.reset()
            except Exception:
                pass
        self.index = None
        self.chunks.clear()
        if hasattr(self, "_fallback_embeddings"):
            try:
                del self._fallback_embeddings
            except AttributeError:
                pass


    def build_index(self, chunks: List[TextChunk], embeddings: Any) -> None:
        """Populate the FAISS index with normalized embeddings and chunk metadata.
        
        Args:
            chunks: List of TextChunk objects.
            embeddings: Numpy array of shape (N, dimension).
        """
        if len(chunks) == 0:
            raise ValueError("Cannot build vector index with 0 chunks.")

        try:
            import numpy as np
            import faiss
            
            # Ensure float32 array
            emb_array = np.asarray(embeddings, dtype=np.float32)
            if emb_array.shape[0] != len(chunks):
                raise ValueError(
                    f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({emb_array.shape[0]})."
                )
            if emb_array.shape[1] != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension}, but got {emb_array.shape[1]}."
                )

            # IndexFlatIP calculates inner products. On L2-normalized vectors, inner product == cosine similarity.
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(emb_array)
            self.chunks = list(chunks)

        except ImportError:
            # Fallback for testing environments without compiled faiss-cpu binaries
            self._build_fallback_index(chunks, embeddings)

    def _build_fallback_index(self, chunks: List[TextChunk], embeddings: Any) -> None:
        """Lightweight in-memory numpy or python fallback for testing without FAISS binary."""
        self.chunks = list(chunks)
        self._fallback_embeddings = embeddings
        self.index = "fallback"

    def search(self, query_vector: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for the top-k most relevant chunks.
        
        Args:
            query_vector: 2D array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of matching chunk dictionaries with score, page, chunk_id, and text.
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        actual_k = min(max(1, top_k), len(self.chunks))

        # Check if FAISS is available
        if self.index != "fallback":
            try:
                import numpy as np
                q_array = np.asarray(query_vector, dtype=np.float32)
                if q_array.ndim == 1:
                    q_array = np.expand_dims(q_array, axis=0)

                scores, indices = self.index.search(q_array, actual_k)
                
                results: List[Dict[str, Any]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    chunk = self.chunks[idx]
                    sim = max(0.0, min(1.0, float(score)))
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page,
                        "text": chunk.text,
                        "score": round(sim, 4),
                        "char_count": chunk.char_count,
                        "word_count": chunk.word_count,
                    })
                return results
            except Exception:
                pass

        # Fallback search if index == "fallback"
        return self._search_fallback(query_vector, actual_k)

    def _search_fallback(self, query_vector: Any, top_k: int) -> List[Dict[str, Any]]:
        """Pure-Python dot product similarity search for fallback/mock testing."""
        try:
            # Flatten query vector if needed
            if hasattr(query_vector, "tolist"):
                q_list = query_vector.tolist()
            elif isinstance(query_vector, list):
                q_list = query_vector
            else:
                q_list = list(query_vector)

            if q_list and isinstance(q_list[0], list):
                q_list = q_list[0]

            scored_chunks = []
            for idx, chunk in enumerate(self.chunks):
                emb = self._fallback_embeddings[idx]
                if hasattr(emb, "tolist"):
                    emb_list = emb.tolist()
                elif isinstance(emb, list):
                    emb_list = emb
                else:
                    emb_list = list(emb)

                # Pure Python dot product
                score = sum(float(a) * float(b) for a, b in zip(q_list, emb_list))
                sim = max(0.0, min(1.0, float(score)))
                scored_chunks.append((sim, chunk))

            # Sort descending by score
            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            results = []
            for sim, chunk in scored_chunks[:top_k]:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "text": chunk.text,
                    "score": round(sim, 4),
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                })
            return results
        except Exception:
            return []
