# DocuMind — Intelligent Document Research Assistant

DocuMind is an NLP-powered document research assistant designed to extract, index, retrieve, and summarize insights from large document collections.

This repository currently implements **Phase 4: RAG-based Question Answering using OpenAI**, featuring a decoupled React (Vite) frontend and FastAPI backend.

---

## 🛠 Tech Stack

- **Frontend**: React 18, Vite, JavaScript, Vanilla CSS (Modern dark glassmorphic design)
- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic, PyMuPDF (`fitz`), Python-Multipart
- **NLP & Search**:
  - **Text Chunking**: Sliding-window overlapping chunker with full page provenance
  - **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions, L2-normalized)
  - **Vector Store**: FAISS (`IndexFlatIP` inner-product search for exact cosine similarity)
  - **Session Registry**: In-memory document and index session store (isolated by `document_id`)
- **RAG & Generative AI**:
  - **Provider**: Official OpenAI Python SDK (`openai>=1.0.0`)
  - **Interface**: OpenAI Responses API (`client.responses.create`)
  - **Model**: Configurable (Default: `gpt-4o-mini`)
  - **Guardrails**: Prompt-injection isolation (untrusted excerpt handling) and backend-enforced source citation mapping
- **Test Suite**: Python `unittest` / `pytest`, `httpx` (35 automated tests, zero external API costs)

---

## 📂 Project Structure

```
documind/
├── .gitignore
├── README.md
├── backend/
│   ├── .env.example                       # Environment variables template with placeholders
│   ├── requirements.txt                   # FastAPI, PyMuPDF, Sentence Transformers, FAISS, OpenAI
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_document_upload.py        # Phase 2 PDF upload & extraction tests
│   │   ├── test_phase3_semantic_search.py # Phase 3 chunking, registry & FAISS search tests
│   │   └── test_phase4_rag.py             # Phase 4 RAG, citations, guardrails & provider tests
│   └── app/
│       ├── __init__.py
│       ├── main.py                        # FastAPI app entry point with CORS, root & health
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py                  # Centralized configuration (OpenAI, CORS, timeouts)
│       ├── api/
│       │   ├── __init__.py
│       │   ├── documents.py               # Upload, /index, /search, and /ask endpoint router
│       │   └── routes.py                  # Modular API routing layer (/api/v1)
│       └── services/
│           ├── __init__.py
│           ├── chunking_service.py        # Overlapping text chunking with metadata preservation
│           ├── document_registry.py       # In-memory session registry (volatile per process)
│           ├── embedding_service.py       # Lazy Sentence Transformers (all-MiniLM-L6-v2)
│           ├── openai_service.py          # OpenAI Responses API integration & safe error mapping
│           ├── pdf_service.py             # In-memory PyMuPDF text extraction
│           ├── rag_service.py             # RAG orchestration, citation mapping & guardrails
│           └── vector_store.py            # FAISS IndexFlatIP cosine similarity search
└── frontend/
    ├── .env.example
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx                        # Main dashboard & Phase 4 RAG workspace
        ├── index.css                      # Design system tokens and glassmorphism styling
        ├── App.css                        # Styling for upload, viewer, search bar, RAG Q&A card
        ├── components/
        │   ├── Header.jsx                 # Navigation header with live status badges
        │   ├── DocumentUpload.jsx         # Drag-and-drop PDF upload component
        │   ├── DocumentViewer.jsx         # Extracted text viewer + semantic search + Ask DocuMind
        │   └── HealthCard.jsx             # Interactive API connectivity tester
        └── services/
            └── api.js                     # API client for health, upload, indexing, search, ask
```

---

## ⚙️ Environment Configuration

Backend configuration is loaded strictly from environment variables or a local `.env` file in `backend/`:

1. Copy `.env.example` to `.env`:
   ```bash
   cp backend/.env.example backend/.env
   ```
2. Configure your OpenAI credentials inside `backend/.env`:
   ```env
   # DocuMind Backend Environment Configuration
   APP_NAME=DocuMind API
   ENVIRONMENT=development
   HOST=127.0.0.1
   PORT=8000

   # CORS Allowed Origins
   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

   # OpenAI API Configuration (Server-side only)
   OPENAI_API_KEY=your-actual-openai-api-key
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TIMEOUT_SECONDS=30.0
   ```

> **Security Guarantee**:
> - `OPENAI_API_KEY` is loaded exclusively on the backend server and is never sent to the browser.
> - `backend/.env` is ignored by `.gitignore` and must never be committed to source control.
> - Credentials are never logged to console or exposed in error messages.

---

## 🚀 Setup and Run Instructions

### 1. Backend Setup

Open a terminal and navigate to the backend directory:

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

#### Run Automated Tests (35 Tests, Mocked OpenAI Client)
```bash
python -m unittest discover -s tests
# or using pytest:
pytest tests/
```

#### Start FastAPI Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend server runs at `http://127.0.0.1:8000`.
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 2. Frontend Setup

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🧠 Phase 4 RAG Architecture & Citations

### `POST /api/v1/documents/{document_id}/ask`
Answers natural language questions grounded strictly in retrieved document passages.

#### Request Payload
```json
{
  "question": "What is the primary thesis evaluated in this research?",
  "top_k": 4
}
```

#### Grounding & Citation Architecture
1. **Semantic Context Retrieval**: Queries FAISS to retrieve the top `k` most relevant passages.
2. **Untrusted Input Guardrails**: System instructions explicitly treat document excerpts as untrusted passive data, preventing prompt-injection overrides embedded inside user PDFs.
3. **Strict Backend Citation Mapping**: Citations (`[1]`, `[2]`, etc.) are mapped to actual retrieved chunks in backend code, ensuring page numbers and chunk IDs can never be hallucinated by the model.
4. **Insufficient Evidence Acknowledgment**: When evidence is missing or ambiguous, the model states that evidence is insufficient, and the backend sets `has_sufficient_evidence = false`.

#### Response Payload (`200 OK`)
```json
{
  "document_id": "8f2b7a90-...",
  "question": "What is the primary thesis evaluated in this research?",
  "answer": "The authors propose a dual-encoder retrieval model [1] that reduces query latency by 40% [2].",
  "model_used": "gpt-4o-mini",
  "has_sufficient_evidence": true,
  "sources": [
    {
      "citation_index": 1,
      "chunk_id": "8f2b7a90-..._p2_c1",
      "page": 2,
      "text": "We evaluate a dual-encoder architecture for dense semantic retrieval...",
      "score": 0.9124,
      "char_count": 340,
      "word_count": 52
    },
    {
      "citation_index": 2,
      "chunk_id": "8f2b7a90-..._p5_c4",
      "page": 5,
      "text": "Latency benchmarks demonstrate a 40% reduction in query response time...",
      "score": 0.8652,
      "char_count": 280,
      "word_count": 41
    }
  ]
}
```

### 5. Document Summarization
- **POST** `/api/v1/documents/{document_id}/summarize`
- Generates structured abstractive summaries with safe context budgeting (<12,000 chars) to prevent oversized requests.
- Returns `overview`, `objectives`, `key_points`, `findings` (with source page references), and `conclusion`.
- In Mock Mode (`LLM_PROVIDER=mock`), deterministically analyzes real page text with zero OpenAI calls.

### 6. Research Paper & Document Comparison
- **GET** `/api/v1/documents` — Lists all documents active in the current session.
- **POST** `/api/v1/documents/compare`
  - Body: `{"document_id_a": "<uuid>", "document_id_b": "<uuid>"}`
  - Analyzes both documents side-by-side: compares objectives, methodologies, findings, limitations, similarities, differences, and missing evidence gaps.

---

## 🗺 Modular Roadmap

- [x] **Phase 1: Project Scaffold** — React (Vite) + FastAPI skeleton, CORS, health endpoints.
- [x] **Phase 2: PDF Upload & Text Extraction** — PyMuPDF in-memory parsing, drag-and-drop UI, page-by-page reader, OCR error detection.
- [x] **Phase 3: Semantic Search & FAISS** — Chunking with overlap, Sentence Transformers (`all-MiniLM-L6-v2`), FAISS `IndexFlatIP`, ranked passage retrieval UI.
- [x] **Phase 4: RAG Question Answering** — OpenAI Responses API, grounded evidence synthesis, backend citation mapping, interactive Q&A interface.
- [x] **Phase 5: Document Intelligence** — Structured document summarization, research paper comparison, safe token budgeting, full UI integration, and 63 passing tests.
