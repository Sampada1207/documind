# DocuMind — Intelligent Document Research and Comparison Platform

DocuMind is an enterprise-grade document intelligence and retrieval-augmented generation (RAG) platform. It provides high-precision semantic search, grounded question answering with verified source citations, structured executive summarization, and side-by-side comparative analysis of research documents and technical specifications.

Built on a decoupled architecture featuring a React (Vite) frontend workspace and a high-performance FastAPI backend, DocuMind operates in both offline deterministic Mock Mode (using FAISS vector search with zero third-party API dependencies) and production OpenAI Mode.

---

## Key Capabilities

- **Multi-Page Research Workspace**: State-driven navigation featuring dedicated screens for Dashboard, Document Library, Research AI, Compare Papers, and System Diagnostics with URL hash synchronization and mobile drawer support.
- **Accurate In-Memory Document Ingestion**: Text extraction powered by PyMuPDF with page-level provenance, character budgeting, and scanned document detection.
- **Dense Vector Search**: Sliding-window chunking with Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions) and FAISS `IndexFlatIP` for exact inner-product cosine similarity search.
- **Grounded Question Answering with Strict Citations**: Backend-enforced citation mapping that ties synthesized statements directly to source pages and retrieved chunk identifiers, eliminating hallucinated references.
- **Structured Document Summarization**: Context-budgeted summarization extracting executive overview, objectives, key findings with page numbers, and conclusions.
- **Deep Document & Research Paper Comparison**: Parameter-level evaluation comparing concrete capacities, deployment scopes, capital budgets, timelines, methodologies, maintenance schedules, and constraints across documents.
- **Dual Engine Architecture**: Supports deterministic offline evaluation in Mock Mode with zero external API costs, as well as production OpenAI GPT integration with structured JSON validation.

---

## Technical Stack

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Vanilla CSS Design System with cohesive dark theme, accessible contrast, and visible keyboard focus states
- **Architecture**: Single-page application shell with URL hash routing (`#dashboard`, `#documents`, `#research`, `#compare`, `#settings`), responsive sidebar, and persistent session state

### Backend
- **Framework**: FastAPI, Uvicorn
- **Validation**: Pydantic v2
- **PDF Extraction**: PyMuPDF (`pymupdf`)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Vector Index**: FAISS (`IndexFlatIP`, L2-normalized)
- **LLM Integration**: Official OpenAI Python SDK (`openai>=1.0.0`) with structured responses and fallback handling

---

## Project Structure

```
documind/
├── .gitignore
├── README.md
├── backend/
│   ├── .env.example                       # Environment variables template
│   ├── requirements.txt                   # FastAPI, PyMuPDF, Sentence Transformers, FAISS, OpenAI
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_comparison_quality.py     # Comparison regression, completeness, and parameter tests
│   │   ├── test_document_upload.py        # PDF upload, size limits, and extraction tests
│   │   ├── test_mock_llm.py               # Deterministic Mock LLM evidence extraction tests
│   │   ├── test_phase3_semantic_search.py # Chunking, embedding, and FAISS vector index tests
│   │   ├── test_phase4_rag.py             # RAG orchestration, citation mapping, and guardrail tests
│   │   ├── test_phase5_intelligence.py    # Summarization and comparison service tests
│   │   ├── test_phase6_integration.py     # End-to-end integration and retrieval benchmarks
│   │   ├── test_phase6_optimization.py    # Query embedding cache and resource lifecycle tests
│   │   └── test_phase6_security_audit.py  # Path traversal, injection, and error hygiene tests
│   └── app/
│       ├── __init__.py
│       ├── main.py                        # FastAPI application entry point, CORS, root, and health
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py                  # Centralized application settings
│       ├── api/
│       │   ├── __init__.py
│       │   ├── documents.py               # Document endpoints: upload, index, search, ask, summarize, compare
│       │   └── routes.py                  # API router mounting (/api/v1)
│       └── services/
│           ├── __init__.py
│           ├── chunking_service.py        # Overlapping text chunker with page metadata
│           ├── comparison_service.py      # Research paper and document comparison service
│           ├── document_registry.py       # In-memory document and FAISS index registry
│           ├── embedding_service.py       # Cached Sentence Transformers embedding service
│           ├── mock_llm_service.py        # Deterministic offline answer synthesizer
│           ├── openai_service.py          # OpenAI Responses API integration
│           ├── pdf_service.py             # PyMuPDF text extraction pipeline
│           ├── rag_service.py             # RAG synthesis and citation enforcement
│           ├── summary_service.py         # Structured executive summarization service
│           └── vector_store.py            # FAISS IndexFlatIP search engine
└── frontend/
    ├── .env.example
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx                        # Multi-page shell coordinator with hash routing
        ├── index.css                      # Design system tokens and foundational dark theme
        ├── App.css                        # Shell layout, sidebar, topbar, and workspace styles
        ├── components/
        │   ├── Sidebar.jsx                # Persistent navigation sidebar with responsive drawer
        │   ├── TopBar.jsx                 # Header bar with breadcrumbs, active doc, and health status
        │   ├── DocumentUpload.jsx         # Drag-and-drop PDF upload component
        │   ├── DocumentViewer.jsx         # Document reader, semantic search, Q&A, and summary viewer
        │   ├── CompareDocuments.jsx       # Side-by-side comparative analysis view
        │   ├── Header.jsx                 # Legacy standalone header component
        │   └── HealthCard.jsx             # Interactive backend connectivity and latency tester
        └── services/
            └── api.js                     # Unified API client for all backend endpoints
```

---

## Environment Configuration

Configuration is loaded from environment variables or a local `.env` file in `backend/`:

1. Copy the example configuration file:
   ```bash
   cp backend/.env.example backend/.env
   ```

2. Review configuration parameters in `backend/.env`:
   ```env
   # Core Application Settings
   APP_NAME=DocuMind API
   ENVIRONMENT=development
   HOST=127.0.0.1
   PORT=8000

   # CORS Allowed Origins
   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

   # LLM Provider Selection ('mock' or 'openai')
   LLM_PROVIDER=mock

   # OpenAI API Configuration (Required only when LLM_PROVIDER=openai)
   OPENAI_API_KEY=your-actual-openai-api-key
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TIMEOUT_SECONDS=30.0
   ```

### Security Considerations
- `OPENAI_API_KEY` is loaded strictly on the backend server and is never transmitted to client browsers.
- Local `.env` files are excluded by `.gitignore` and must never be committed to source control.
- Credentials and internal stack traces are sanitized from all client-facing error responses.

---

## Installation and Execution

### 1. Backend Service

Prerequisites: Python 3.10 or higher.

Navigate to the `backend` directory, initialize a virtual environment, and install dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run Backend Test Suite
The test suite consists of 94 automated unit, integration, and regression tests operating without external network dependencies:

```bash
pytest tests/
```

#### Start FastAPI Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend server is accessible at `http://127.0.0.1:8000`:
- Interactive Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Service Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 2. Frontend Application

Prerequisites: Node.js 18 or higher.

Navigate to the `frontend` directory, install dependencies, and launch the development server:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

#### Production Build
To create an optimized production build:

```bash
npm run build
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service root and metadata information |
| `GET` | `/health` | Service health status, environment, and active LLM provider |
| `POST` | `/api/v1/documents/upload` | Upload PDF and extract text with page-level provenance |
| `GET` | `/api/v1/documents` | List all active documents registered in memory |
| `GET` | `/api/v1/documents/{id}` | Retrieve document status, page count, and indexing state |
| `POST` | `/api/v1/documents/{id}/index` | Chunk document and construct in-memory FAISS index |
| `POST` | `/api/v1/documents/{id}/search` | Semantic vector search across document chunks |
| `POST` | `/api/v1/documents/{id}/ask` | Grounded RAG question answering with verified citations |
| `POST` | `/api/v1/documents/{id}/summarize` | Generate structured executive summary with page references |
| `POST` | `/api/v1/documents/compare` | Side-by-side comparative analysis of two documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete document, release vector index, and clear cache |

---

## Architecture and Design Principles

1. **Grounded Provenance**: Statements generated by the platform include verified document and page references derived from physical document chunks.
2. **Context Budgeting**: Context windows are monitored and budgeted (<12,000 characters for summaries and comparisons) to eliminate oversized payload errors.
3. **Defensive Processing**: PDF line wraps are normalized into coherent sentences, preserving headers and numerical scopes without generating fragments or filler text.
4. **Volatile Session Isolation**: Documents and vector indices reside in an in-memory session registry isolated by unique identifiers.
5. **Zero-Cost Verification**: Full system validation is executable offline in Mock Mode, ensuring continuous testing without recurring API fees.

---

## License

This project is licensed under the MIT License.
