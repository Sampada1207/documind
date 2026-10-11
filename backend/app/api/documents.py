"""Document upload, indexing, semantic search, and RAG Q&A endpoints."""

import logging
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.services.chunking_service import chunk_document
from app.services.document_registry import registry
from app.services.embedding_service import DEFAULT_MODEL_NAME, EMBEDDING_DIMENSION, embedding_service
from app.services.openai_service import (
    OpenAIAuthenticationError,
    OpenAIConfigurationError,
    OpenAINetworkError,
    OpenAIProviderError,
    OpenAIRateLimitError,
)
from app.services.pdf_service import (
    MAX_PDF_SIZE_BYTES,
    EmptyPDFError,
    InvalidPDFError,
    PDFSizeLimitExceededError,
    ScannedPDFError,
    extract_text_from_pdf,
)
from app.services.rag_service import run_rag_pipeline
from app.services.summary_service import summary_service
from app.services.comparison_service import comparison_service
from app.services.vector_store import FAISSVectorStore

import os
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Non-deprecated status codes with backwards-compatible integer fallback
HTTP_413_CONTENT_TOO_LARGE = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
HTTP_422_UNPROCESSABLE_CONTENT = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


DOC_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def sanitize_filename(raw_name: Optional[str]) -> str:
    """Sanitize uploaded filename to prevent directory traversal and script injection."""
    if not raw_name or not raw_name.strip():
        return "uploaded.pdf"
    # Extract only the base name (prevents directory traversal e.g. ../../)
    base = os.path.basename(raw_name.strip())
    # Replace control characters, null bytes, and script/tag delimiters
    cleaned = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', '_', base)
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        return "uploaded.pdf"
    return cleaned[:255]



def validate_document_id(doc_id: str) -> str:
    """Validate document ID format to prevent injection and path traversal."""
    if not doc_id or not DOC_ID_REGEX.match(doc_id.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document identifier format.",
        )
    return doc_id.strip()


# --- Schemas ---

class PageExtractResponse(BaseModel):
    page: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Extracted text content from this page")
    char_count: int = Field(..., description="Number of characters on this page")
    word_count: int = Field(..., description="Number of words on this page")


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID identifier")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    total_pages: int = Field(..., description="Total number of pages in the PDF")
    total_characters: int = Field(..., description="Total characters extracted across all pages")
    total_words: int = Field(..., description="Total words extracted across all pages")
    pages: List[PageExtractResponse] = Field(..., description="List of pages with extracted text")
    is_indexed: bool = Field(False, description="Whether this document has been indexed in FAISS")


class DocumentIndexRequest(BaseModel):
    chunk_size: int = Field(500, ge=100, le=2000, description="Target character length per chunk")
    chunk_overlap: int = Field(100, ge=0, le=500, description="Overlapping character length between chunks")


class DocumentIndexResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Source filename")
    status: str = Field("indexed", description="Indexing operational status")
    total_chunks: int = Field(..., description="Total number of text chunks created")
    embedding_model: str = Field(..., description="Model used for embeddings")
    embedding_dimension: int = Field(..., description="Dimensionality of embeddings")
    message: str = Field(..., description="Status summary message")


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of top matching passages to return")


class SearchResultItem(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    page: int = Field(..., description="1-indexed source page number")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    char_count: int = Field(..., description="Character count in passage")
    word_count: int = Field(..., description="Word count in passage")


class SearchResultResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier searched")
    query: str = Field(..., description="User query executed")
    total_results: int = Field(..., description="Number of matching passages found")
    results: List[SearchResultItem] = Field(..., description="Ranked list of relevant passages")


class AskQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the document content")
    top_k: int = Field(4, ge=1, le=10, description="Number of context passages to retrieve")


class SourceExcerpt(BaseModel):
    citation_index: int = Field(..., description="Citation marker number [1], [2], etc.")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    filename: Optional[str] = Field(None, description="Source document filename")
    page: int = Field(..., description="1-indexed source page number")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Cosine retrieval similarity score")
    char_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")


class AskQuestionResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    question: str = Field(..., description="User question asked")
    answer: str = Field(..., description="Synthesized grounded answer")
    model_used: str = Field(..., description="Model or engine utilized")
    llm_provider: str = Field(..., description="Active LLM provider: 'mock' or 'openai'")
    has_sufficient_evidence: bool = Field(..., description="Whether evidence was sufficient")
    sources: List[SourceExcerpt] = Field(..., description="Verified backend-mapped source citations")


class FindingItem(BaseModel):
    text: str = Field(..., description="Extracted finding statement")
    page: int = Field(..., description="Source page number")


class DocumentSummaryResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Source document filename")
    total_pages: int = Field(..., description="Total pages in document")
    overview: str = Field(..., description="High-level document overview")
    objectives: List[str] = Field(default_factory=list, description="Key objectives with page references")
    key_points: List[str] = Field(default_factory=list, description="Core takeaways")
    findings: List[FindingItem] = Field(default_factory=list, description="Concrete factual findings with page numbers")
    conclusion: str = Field(..., description="Document conclusion")
    llm_provider: str = Field(..., description="Active LLM provider: 'mock' or 'openai'")
    is_simulated: bool = Field(..., description="Whether summary was simulated (mock mode)")
    model_used: str = Field(..., description="Model identifier utilized")
    sampled_pages_count: int = Field(..., description="Number of sampled pages utilized for summary")


class DocumentCompareRequest(BaseModel):
    document_id_a: str = Field(..., min_length=1, description="First document UUID to compare")
    document_id_b: str = Field(..., min_length=1, description="Second document UUID to compare")


class DocumentComparisonProfile(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    total_pages: int = Field(..., description="Total pages in document")
    objectives: List[str] = Field(default_factory=list, description="Document objectives with page references")
    methodologies: List[str] = Field(default_factory=list, description="Methodologies and approaches")
    findings: List[str] = Field(default_factory=list, description="Document-specific key findings")
    limitations: List[str] = Field(default_factory=list, description="Identified limitations or constraints")


class ComparisonCitation(BaseModel):
    document_id: str = Field(..., description="Source document UUID")
    filename: str = Field(..., description="Source document filename")
    page: int = Field(..., description="Page number")
    excerpt: str = Field(..., description="Excerpt statement")


class DocumentCompareResponse(BaseModel):
    document_a: DocumentComparisonProfile = Field(..., description="Profile of Document A")
    document_b: DocumentComparisonProfile = Field(..., description="Profile of Document B")
    similarities: List[str] = Field(default_factory=list, description="Shared technical aspects or conclusions")
    differences: List[str] = Field(default_factory=list, description="Contrasting scopes, methodologies, or findings")
    missing_information: List[str] = Field(default_factory=list, description="Gaps or missing details in either document")
    citations: List[ComparisonCitation] = Field(default_factory=list, description="Citations supporting comparison points")
    llm_provider: str = Field(..., description="Active LLM provider: 'mock' or 'openai'")
    is_simulated: bool = Field(..., description="Whether comparison was simulated (mock mode)")
    model_used: str = Field(..., description="Model identifier utilized")


class DocumentSummaryInfo(BaseModel):
    document_id: str = Field(..., description="Document UUID")
    filename: str = Field(..., description="Document filename")
    total_pages: int = Field(..., description="Total page count")
    is_indexed: bool = Field(..., description="Whether FAISS index is built")
    chunk_count: int = Field(..., description="Total chunk count")


# --- Endpoints ---

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and extract text from a PDF document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and extract text from"),
):
    """Upload a PDF file, validate it, extract text, and register in the in-memory session."""
    raw_name = file.filename or "unknown.pdf"
    filename = sanitize_filename(raw_name)

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported.",
        )

    if file.content_type and file.content_type.lower() not in (
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid MIME type. Uploaded file must be an application/pdf.",
        )

    try:
        content = await file.read(MAX_PDF_SIZE_BYTES + 1)
    except Exception as e:
        logger.error(f"Failed to read uploaded stream for {filename}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content.",
        )

    if len(content) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty (0 bytes).",
        )

    try:
        extraction_result = extract_text_from_pdf(content, filename)
        registry.register_document(extraction_result)
        extraction_result["is_indexed"] = False
        return extraction_result
    except EmptyPDFError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PDFSizeLimitExceededError as e:
        raise HTTPException(status_code=HTTP_413_CONTENT_TOO_LARGE, detail=str(e))
    except InvalidPDFError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ScannedPDFError as e:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server text extraction engine is not available. Please verify server dependencies.",
        )
    except Exception as e:
        logger.error(f"Unexpected error processing document {filename}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the PDF.",
        )


@router.post(
    "/{document_id}/index",
    response_model=DocumentIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk and index document text using Sentence Transformers and FAISS",
)
def index_document(
    document_id: str,
    index_params: Optional[DocumentIndexRequest] = None,
):
    """Chunk the extracted text of an active document and build an in-memory FAISS index."""
    valid_id = validate_document_id(document_id)
    doc = registry.get_document(valid_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{valid_id}' not found in in-memory session. Please re-upload the document.",
        )

    chunk_size = index_params.chunk_size if index_params else 500
    chunk_overlap = index_params.chunk_overlap if index_params else 100

    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be strictly less than chunk_size.",
        )

    try:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        logger.error(f"Failed to chunk document {document_id}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to chunk document text.",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document contains no extractable text to chunk and index.",
        )

    chunk_texts = [c.text for c in chunks]
    try:
        embeddings = embedding_service.embed_texts(chunk_texts)
    except RuntimeError as e:
        logger.error(f"Embedding service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected embedding error for {document_id}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings for document chunks.",
        )

    try:
        vector_store = FAISSVectorStore(dimension=embedding_service.dimension)
        vector_store.build_index(chunks, embeddings)
        registry.set_index(document_id, vector_store, len(chunks))
    except Exception as e:
        logger.error(f"Failed to build vector index for {document_id}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize and populate vector store.",
        )

    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "status": "indexed",
        "total_chunks": len(chunks),
        "embedding_model": embedding_service.model_name,
        "embedding_dimension": embedding_service.dimension,
        "message": f"Successfully indexed {len(chunks)} chunks using {embedding_service.model_name}.",
    }


@router.post(
    "/{document_id}/search",
    response_model=SearchResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic similarity search over indexed document chunks",
)
def search_document(
    document_id: str,
    search_params: SearchQueryRequest,
):
    """Execute natural language cosine-similarity search against indexed chunks."""
    valid_id = validate_document_id(document_id)
    query = search_params.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace only.",
        )

    doc = registry.get_document(valid_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{valid_id}' not found in in-memory session. Please re-upload the document.",
        )

    if not registry.is_indexed(valid_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{valid_id}' has not been indexed yet. Please trigger indexing before searching.",
        )

    vector_store = registry.get_index(valid_id)
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector store could not be retrieved from active session.",
        )

    try:
        query_vector = embedding_service.embed_query(query)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to encode query: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode search query.",
        )

    try:
        results = vector_store.search(query_vector, top_k=search_params.top_k)
    except Exception as e:
        logger.error(f"Search failed on document {document_id}: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during vector similarity search.",
        )

    return {
        "document_id": document_id,
        "query": query,
        "total_results": len(results),
        "results": results,
    }


@router.post(
    "/{document_id}/ask",
    response_model=AskQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the document using RAG and OpenAI",
)
def ask_document_question(
    document_id: str,
    ask_params: AskQuestionRequest,
):
    """Answer user question using retrieved FAISS excerpts and OpenAI Responses API."""
    valid_id = validate_document_id(document_id)
    question = ask_params.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        result = run_rag_pipeline(
            document_id=valid_id,
            question=question,
            top_k=ask_params.top_k,
        )
        return result

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except OpenAIConfigurationError as e:
        logger.error("RAG execution failed: OpenAI key not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in the backend environment.",
        )
    except OpenAIAuthenticationError as e:
        logger.error("RAG execution failed: OpenAI authentication error.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI authentication failed. Please verify your OPENAI_API_KEY.",
        )
    except OpenAIRateLimitError as e:
        logger.warning("RAG execution failed: OpenAI rate limit.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit or usage quota exceeded. Please try again shortly.",
        )
    except OpenAINetworkError as e:
        logger.error("RAG execution failed: Network timeout.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Connection to OpenAI timed out or failed. Please check network connectivity.",
        )
    except OpenAIProviderError as e:
        logger.error(f"RAG execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI service returned an error while synthesizing answer.",
        )
    except Exception as e:
        logger.error(f"Unexpected RAG error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the question.",
        )


@router.get(
    "/{document_id}",
    summary="Get document session metadata and indexing status",
)
def get_document_status(document_id: str):
    """Retrieve metadata and index status for a document."""
    valid_id = validate_document_id(document_id)
    doc = registry.get_document(valid_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{valid_id}' not found in in-memory session.",
        )
    return {
        "document_id": doc["document_id"],
        "filename": doc["filename"],
        "total_pages": doc["total_pages"],
        "total_characters": doc["total_characters"],
        "total_words": doc["total_words"],
        "is_indexed": doc.get("is_indexed", False),
        "chunk_count": doc.get("chunk_count", 0),
    }


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an uploaded document and its vector index from in-memory session",
)
def delete_document(document_id: str):
    """Remove a document and its associated FAISS vector index from memory."""
    valid_id = validate_document_id(document_id)
    existed = registry.remove_document(valid_id)
    if not existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{valid_id}' not found in in-memory session.",
        )
    return {
        "document_id": valid_id,
        "message": f"Document '{valid_id}' and all associated indices successfully removed from memory.",
    }


@router.get(
    "",
    response_model=List[DocumentSummaryInfo],
    summary="List all uploaded documents in the active session",
)
def list_documents():
    """Retrieve all uploaded documents in the in-memory session."""
    return registry.list_documents()


@router.post(
    "/{document_id}/summarize",
    response_model=DocumentSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a structured summary for an uploaded document",
)
def summarize_document(document_id: str):
    """Summarize document with overview, objectives, key points, findings with page citations, and conclusion."""
    valid_id = validate_document_id(document_id)
    try:
        return summary_service.summarize_document(valid_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except OpenAIConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in the backend environment.",
        )
    except OpenAIAuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI authentication failed. Please verify your OPENAI_API_KEY.",
        )
    except OpenAIRateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit or usage quota exceeded. Please try again shortly.",
        )
    except OpenAINetworkError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Connection to OpenAI timed out or failed. Please check network connectivity.",
        )
    except OpenAIProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI service returned an error while summarizing: {e}",
        )
    except Exception as e:
        logger.error(f"Summarization failed for document {valid_id}: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the document summary.",
        )


@router.post(
    "/compare",
    response_model=DocumentCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare two uploaded research papers or documents",
)
def compare_documents(request: DocumentCompareRequest):
    """Compare objectives, methodologies, findings, limitations, similarities, and differences between two documents."""
    doc_id_a = validate_document_id(request.document_id_a)
    doc_id_b = validate_document_id(request.document_id_b)
    try:
        return comparison_service.compare_documents(doc_id_a, doc_id_b)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except OpenAIConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in the backend environment.",
        )
    except OpenAIAuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI authentication failed. Please verify your OPENAI_API_KEY.",
        )
    except OpenAIRateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit or usage quota exceeded. Please try again shortly.",
        )
    except OpenAINetworkError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Connection to OpenAI timed out or failed. Please check network connectivity.",
        )
    except OpenAIProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI service returned an error while comparing: {e}",
        )
    except Exception as e:
        logger.error(f"Comparison failed between {doc_id_a} and {doc_id_b}: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while comparing the documents.",
        )

