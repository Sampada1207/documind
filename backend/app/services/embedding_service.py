import collections
import time
from typing import Any, Dict, List, Optional

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_QUERY_CACHE_SIZE = 128


class QueryEmbeddingCache:
    """Thread-safe bounded LRU cache for repeated query embeddings to eliminate redundant CPU encoding."""

    def __init__(self, max_size: int = DEFAULT_QUERY_CACHE_SIZE):
        self.max_size = max(1, max_size)
        # Maps normalized_query -> (embedding_vector, timestamp)
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def get(self, query: str) -> Optional[Any]:
        """Retrieve cached embedding for normalized query, moving it to MRU position."""
        key = query.strip().lower()
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            vector, _ = self._cache[key]
            return vector.copy() if hasattr(vector, "copy") else vector
        self.misses += 1
        return None

    def put(self, query: str, vector: Any) -> None:
        """Store query embedding in cache, enforcing strict LRU eviction if capacity is reached."""
        key = query.strip().lower()
        # Security: Do not cache excessively long strings (>500 chars) to prevent memory ballooning
        if len(key) > 500:
            return

        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Evict least-recently-used item (front of OrderedDict)
                self._cache.popitem(last=False)
                self.evictions += 1

        stored_vec = vector.copy() if hasattr(vector, "copy") else vector
        self._cache[key] = (stored_vec, time.time())

    def clear(self) -> None:
        """Clear all cached embeddings and reset metrics."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self.hits + self.misses
        hit_ratio = round((self.hits / total), 4) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_ratio": hit_ratio,
        }


class EmbeddingService:
    """Manages embedding generation with lazy model loading, normalization, and LRU query caching."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self._model = None
        self.query_cache = QueryEmbeddingCache(max_size=DEFAULT_QUERY_CACHE_SIZE)

    def _load_model(self):
        """Lazy load SentenceTransformer model on first inference request."""
        if self._model is not None:
            return self._model

        try:
            import numpy as np  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Embedding dependencies are not installed. "
                "Please run: pip install -r requirements.txt (sentence-transformers, numpy)"
            ) from e

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def set_mock_model(self, mock_model) -> None:
        """Inject a mock model for isolated testing without network downloads."""
        self._model = mock_model

    def clear_cache(self) -> None:
        """Clear query embedding cache."""
        self.query_cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Retrieve current query embedding cache metrics."""
        return self.query_cache.stats()

    def embed_texts(self, texts: List[str]):
        """Generate L2-normalized embeddings for a list of texts.
        
        Args:
            texts: List of strings to encode.
            
        Returns:
            np.ndarray of shape (len(texts), 384) with dtype float32.
        """
        if not texts:
            try:
                import numpy as np
                return np.empty((0, self.dimension), dtype=np.float32)
            except ImportError:
                return []

        model = self._load_model()
        # encode with normalize_embeddings=True for cosine similarity via inner product
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        try:
            import numpy as np
            if isinstance(embeddings, np.ndarray):
                return embeddings.astype(np.float32)
        except ImportError:
            pass

        return embeddings

    def embed_query(self, query: str):
        """Generate or retrieve cached normalized embedding for a single natural language query.
        
        Args:
            query: User search query string.
            
        Returns:
            np.ndarray of shape (1, 384) with dtype float32.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        cached = self.query_cache.get(clean_query)
        if cached is not None:
            return cached

        vector = self.embed_texts([clean_query])
        self.query_cache.put(clean_query, vector)
        return vector


# Global service instance
embedding_service = EmbeddingService()

