"""Service layer for DocuMind.

Exposes:
- extract_text_from_pdf: In-memory text extraction from PDF documents using PyMuPDF.
- chunk_document, chunk_text: Sliding-window overlapping text chunking.
- embedding_service: Sentence Transformers embedding generation (all-MiniLM-L6-v2).
- FAISSVectorStore: Cosine similarity vector search over FAISS IndexFlatIP.
- registry: In-memory document session registry.
- openai_service: OpenAI Responses API provider service.
- run_rag_pipeline: RAG pipeline with grounded question answering and citation mapping.
"""

from app.services.chunking_service import TextChunk, chunk_document, chunk_text
from app.services.document_registry import DocumentRegistry, registry
from app.services.embedding_service import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EmbeddingService,
    embedding_service,
)
from app.services.openai_service import (
    OpenAIAuthenticationError,
    OpenAIConfigurationError,
    OpenAINetworkError,
    OpenAIProviderError,
    OpenAIRateLimitError,
    OpenAIService,
    openai_service,
)
from app.services.pdf_service import (
    MAX_PDF_SIZE_BYTES,
    EmptyPDFError,
    InvalidPDFError,
    PDFProcessingError,
    PDFSizeLimitExceededError,
    ScannedPDFError,
    extract_text_from_pdf,
)
from app.services.rag_service import format_context_prompt, run_rag_pipeline
from app.services.vector_store import FAISSVectorStore

__all__ = [
    "MAX_PDF_SIZE_BYTES",
    "EmptyPDFError",
    "InvalidPDFError",
    "PDFProcessingError",
    "PDFSizeLimitExceededError",
    "ScannedPDFError",
    "extract_text_from_pdf",
    "TextChunk",
    "chunk_document",
    "chunk_text",
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "EmbeddingService",
    "embedding_service",
    "FAISSVectorStore",
    "DocumentRegistry",
    "registry",
    "OpenAIService",
    "openai_service",
    "OpenAIConfigurationError",
    "OpenAIAuthenticationError",
    "OpenAIRateLimitError",
    "OpenAINetworkError",
    "OpenAIProviderError",
    "run_rag_pipeline",
    "format_context_prompt",
]
