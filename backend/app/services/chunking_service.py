"""Text chunking service for document tokenization with metadata preservation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class TextChunk:
    """Represents a text chunk with complete provenance metadata."""
    chunk_id: str
    document_id: str
    source_filename: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return asdict(self)


def chunk_text(
    text: str,
    page_number: int,
    document_id: str,
    source_filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    start_chunk_index: int = 0,
) -> List[TextChunk]:
    """Split text from a single page into overlapping chunks.
    
    Args:
        text: Raw page text content.
        page_number: 1-indexed page number from source document.
        document_id: Unique document identifier.
        source_filename: Source PDF filename.
        chunk_size: Maximum character length per chunk (default: 500).
        chunk_overlap: Number of characters to overlap between adjacent chunks (default: 100).
        start_chunk_index: Starting counter for chunk index.
        
    Returns:
        List of TextChunk instances preserving all metadata and overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # If the text is smaller than or equal to the chunk size, return it as a single chunk
    if len(cleaned_text) <= chunk_size:
        words = cleaned_text.split()
        return [
            TextChunk(
                chunk_id=f"{document_id}_p{page_number}_c{start_chunk_index}",
                document_id=document_id,
                source_filename=source_filename,
                page=page_number,
                chunk_index=start_chunk_index,
                text=cleaned_text,
                char_count=len(cleaned_text),
                word_count=len(words),
            )
        ]

    step = chunk_size - chunk_overlap
    chunks: List[TextChunk] = []
    chunk_idx = start_chunk_index
    start_idx = 0
    total_len = len(cleaned_text)

    while start_idx < total_len:
        end_idx = min(start_idx + chunk_size, total_len)
        raw_slice = cleaned_text[start_idx:end_idx].strip()

        if raw_slice:
            words = raw_slice.split()
            chunk = TextChunk(
                chunk_id=f"{document_id}_p{page_number}_c{chunk_idx}",
                document_id=document_id,
                source_filename=source_filename,
                page=page_number,
                chunk_index=chunk_idx,
                text=raw_slice,
                char_count=len(raw_slice),
                word_count=len(words),
            )
            chunks.append(chunk)
            chunk_idx += 1

        if end_idx >= total_len:
            break

        start_idx += step

    return chunks


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[TextChunk]:
    """Chunk all pages of an extracted document, preserving page-level provenance.
    
    Args:
        document: Document dictionary from extraction/registry.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap characters between chunks.
        
    Returns:
        List of TextChunk instances across all pages.
    """
    doc_id = document["document_id"]
    filename = document["filename"]
    pages = document.get("pages", [])

    all_chunks: List[TextChunk] = []
    chunk_counter = 0

    for page_data in pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        
        page_chunks = chunk_text(
            text=page_text,
            page_number=page_num,
            document_id=doc_id,
            source_filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=chunk_counter,
        )
        all_chunks.extend(page_chunks)
        chunk_counter += len(page_chunks)

    return all_chunks
