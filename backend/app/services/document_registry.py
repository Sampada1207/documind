"""In-memory document session registry.

Lifecycle and Architecture Note:
--------------------------------
This in-memory registry isolates uploaded documents and their corresponding
FAISS vector search indices by `document_id`.
Data stored here is ephemeral and will be cleared whenever the backend process
restarts. No disk storage or external databases are introduced in this phase.
"""

from typing import Any, Dict, List, Optional


# Maximum in-memory document capacity to mitigate unbounded memory growth / DoS
MAX_REGISTERED_DOCUMENTS = 100


class DocumentRegistry:
    """Thread-safe in-memory registry holding active document sessions and FAISS indices."""

    def __init__(self, max_capacity: int = MAX_REGISTERED_DOCUMENTS) -> None:
        self.max_capacity = max_capacity
        # Maps document_id -> document metadata & pages
        self._documents: Dict[str, Dict[str, Any]] = {}
        # Maps document_id -> VectorStore instance
        self._indices: Dict[str, Any] = {}

    def register_document(self, document_data: Dict[str, Any]) -> None:
        """Register an uploaded document in the in-memory session store."""
        doc_id = document_data["document_id"]

        # Evict oldest registered document if capacity reached (FIFO mitigation for memory exhaustion)
        if len(self._documents) >= self.max_capacity and doc_id not in self._documents:
            oldest_id = next(iter(self._documents))
            self.remove_document(oldest_id)

        self._documents[doc_id] = {
            "document_id": doc_id,
            "filename": document_data["filename"],
            "total_pages": document_data["total_pages"],
            "total_characters": document_data["total_characters"],
            "total_words": document_data["total_words"],
            "pages": document_data["pages"],
            "is_indexed": False,
            "chunk_count": 0,
        }


    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve registered document metadata and pages by document ID."""
        return self._documents.get(document_id)

    def set_index(self, document_id: str, vector_store: Any, chunk_count: int) -> None:
        """Associate an indexed vector store with a registered document."""
        if document_id in self._documents:
            self._documents[document_id]["is_indexed"] = True
            self._documents[document_id]["chunk_count"] = chunk_count
        self._indices[document_id] = vector_store

    def get_index(self, document_id: str) -> Optional[Any]:
        """Retrieve the active vector store for a document ID."""
        return self._indices.get(document_id)

    def is_indexed(self, document_id: str) -> bool:
        """Check if a document exists and has a populated vector index."""
        doc = self._documents.get(document_id)
        return bool(doc and doc.get("is_indexed") and document_id in self._indices)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List summary of all currently registered documents in memory."""
        return [
            {
                "document_id": d["document_id"],
                "filename": d["filename"],
                "total_pages": d["total_pages"],
                "is_indexed": d.get("is_indexed", False),
                "chunk_count": d.get("chunk_count", 0),
            }
            for d in self._documents.values()
        ]

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and explicitly release its FAISS index and chunk references."""
        doc_exists = document_id in self._documents
        self._documents.pop(document_id, None)
        index = self._indices.pop(document_id, None)
        if index is not None and hasattr(index, "clear"):
            try:
                index.clear()
            except Exception:
                pass
        return doc_exists

    def clear(self) -> None:
        """Clear all in-memory documents and release all associated index resources."""
        for index in list(self._indices.values()):
            if index is not None and hasattr(index, "clear"):
                try:
                    index.clear()
                except Exception:
                    pass
        self._documents.clear()
        self._indices.clear()



# Global singleton instance
registry = DocumentRegistry()
