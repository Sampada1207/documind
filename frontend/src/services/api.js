const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Extract human-readable error messages from API responses, including Pydantic 422 arrays
 */
function formatApiErrorMessage(responseData, defaultMessage) {
  if (responseData && responseData.detail) {
    if (typeof responseData.detail === 'string') {
      return responseData.detail;
    }
    if (Array.isArray(responseData.detail)) {
      return responseData.detail
        .map((item) => (item.msg ? `${item.loc ? item.loc.slice(-1) + ': ' : ''}${item.msg}` : JSON.stringify(item)))
        .join('; ');
    }
    if (typeof responseData.detail === 'object') {
      return JSON.stringify(responseData.detail);
    }
  }
  return defaultMessage;
}

/**
 * Fetch application info from root endpoint
 */
export async function fetchRootInfo() {
  const startTime = performance.now();
  const response = await fetch(`${API_BASE_URL}/`, {
    headers: { 'Content-Type': 'application/json' },
  });
  const latency = Math.round(performance.now() - startTime);

  if (!response.ok) {
    throw new Error(`API returned status ${response.status}: ${response.statusText}`);
  }

  const data = await response.json();
  return { data, latency };
}

/**
 * Fetch backend health status
 */
export async function checkBackendHealth() {
  const startTime = performance.now();
  const response = await fetch(`${API_BASE_URL}/health`, {
    headers: { 'Content-Type': 'application/json' },
  });
  const latency = Math.round(performance.now() - startTime);

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }

  const data = await response.json();
  return { data, latency };
}

/**
 * Upload a PDF document for in-memory text extraction
 * @param {File} file - The PDF file object
 * @returns {Promise<Object>} Extracted document structure with pages
 */
export async function uploadPdfDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg = formatApiErrorMessage(
      responseData,
      `Upload failed with HTTP ${response.status}: ${response.statusText}`
    );
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * Trigger in-memory chunking and FAISS indexing for an uploaded document
 * @param {string} documentId - The document UUID
 * @param {Object} options - chunk size and overlap configuration
 * @returns {Promise<Object>} Indexing response with total chunks created
 */
export async function indexDocument(documentId, options = { chunkSize: 500, chunkOverlap: 100 }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chunk_size: options.chunkSize || 500,
      chunk_overlap: options.chunkOverlap || 100,
    }),
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg =
      (isJson && responseData && responseData.detail) ||
      `Indexing failed with HTTP ${response.status}: ${response.statusText}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * Perform natural language semantic search across indexed document chunks
 * @param {string} documentId - The document UUID
 * @param {string} query - Natural language search query
 * @param {number} topK - Number of top chunks to retrieve
 * @returns {Promise<Object>} Search results with ranked passages and similarity scores
 */
export async function searchDocument(documentId, query, topK = 5) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query.trim(),
      top_k: topK,
    }),
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg =
      (isJson && responseData && responseData.detail) ||
      `Search failed with HTTP ${response.status}: ${response.statusText}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * Retrieve document status and indexing state
 * @param {string} documentId - The document UUID
 * @returns {Promise<Object>} Document status metadata
 */
export async function getDocumentStatus(documentId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Failed to get document status: HTTP ${response.status}`);
  }

  return await response.json();
}

/**
 * Ask a question about the document using RAG (OpenAI API + FAISS retrieval)
 * @param {string} documentId - The document UUID
 * @param {string} question - The user question
 * @param {number} topK - Number of evidence passages to retrieve
 * @returns {Promise<Object>} Synthesized answer with citations and evidence sources
 */
export async function askDocumentQuestion(documentId, question, topK = 4) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: question.trim(),
      top_k: topK,
    }),
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg =
      (isJson && responseData && responseData.detail) ||
      `Question answering failed with HTTP ${response.status}: ${response.statusText}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * List all uploaded documents in the active session
 * @returns {Promise<Array>} List of document metadata summaries
 */
export async function listUploadedDocuments() {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Failed to list documents: HTTP ${response.status}`);
  }

  return await response.json();
}

/**
 * Generate a structured summary for an uploaded document
 * @param {string} documentId - The document UUID
 * @returns {Promise<Object>} Structured summary response
 */
export async function summarizeDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg =
      (isJson && responseData && responseData.detail) ||
      `Summarization failed with HTTP ${response.status}: ${response.statusText}`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * Compare two uploaded documents/research papers
 * @param {string} documentIdA - First document UUID
 * @param {string} documentIdB - Second document UUID
 * @returns {Promise<Object>} Comparative analysis response
 */
export async function compareDocuments(documentIdA, documentIdB) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      document_id_a: documentIdA,
      document_id_b: documentIdB,
    }),
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg = formatApiErrorMessage(
      responseData,
      `Comparison failed with HTTP ${response.status}: ${response.statusText}`
    );
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}

/**
 * Delete an uploaded document and its vector index from in-memory session
 * @param {string} documentId - The document UUID
 * @returns {Promise<Object>} Deletion confirmation response
 */
export async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
  });

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const responseData = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorMsg = formatApiErrorMessage(
      responseData,
      `Delete failed with HTTP ${response.status}: ${response.statusText}`
    );
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = responseData;
    throw err;
  }

  return responseData;
}
