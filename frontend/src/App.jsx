import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import HealthCard from './components/HealthCard';
import DocumentUpload from './components/DocumentUpload';
import DocumentViewer from './components/DocumentViewer';
import CompareDocuments from './components/CompareDocuments';
import { checkBackendHealth } from './services/api';
import './App.css';

export default function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [extractedDocument, setExtractedDocument] = useState(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [showDirectComparison, setShowDirectComparison] = useState(false);

  useEffect(() => {
    // Initial health check on page load
    checkBackendHealth()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  return (
    <div className="app-container">
      <Header backendStatus={backendStatus} />

      {/* Main Hero & Phase 5 Feature Area */}
      <section className="hero-card">
        <div className="hero-text-container">
          <h2 className="hero-title">Intelligent Document Research Assistant</h2>
          <p className="hero-description">
            Grounded RAG question answering, structured abstractive summaries with page citations, and
            deep side-by-side research paper comparison powered by FAISS semantic search and LLM intelligence.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            className="btn-toggle-diagnostics"
            onClick={() => setShowDirectComparison((prev) => !prev)}
          >
            {showDirectComparison ? 'Back to Document Workspace' : '⚖️ Compare Research Papers'}
          </button>
          <button
            className="btn-toggle-diagnostics"
            onClick={() => setShowDiagnostics((prev) => !prev)}
          >
            {showDiagnostics ? 'Hide Diagnostics' : 'System Diagnostics'}
          </button>
        </div>
      </section>

      {/* Primary Workspace */}
      <section className="workspace-section">
        {showDirectComparison ? (
          <CompareDocuments
            currentDocumentId={extractedDocument?.document_id}
            onClose={() => setShowDirectComparison(false)}
          />
        ) : !extractedDocument ? (
          <DocumentUpload
            onExtractionSuccess={setExtractedDocument}
            isBackendOnline={backendStatus}
          />
        ) : (
          <DocumentViewer
            documentData={extractedDocument}
            onReset={() => setExtractedDocument(null)}
          />
        )}
      </section>

      {/* Diagnostics & Roadmap Section */}
      {showDiagnostics && (
        <section className="diagnostics-section">
          <div className="dashboard-grid">
            <HealthCard onHealthUpdate={setBackendStatus} />

            <div className="card">
              <div className="card-title">
                <span>Modular Architecture Roadmap</span>
              </div>
              <p className="card-subtitle">
                Component lifecycle and implementation status across development phases.
              </p>

              <div className="arch-list">
                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 1: Project Scaffold</strong>
                    <span style={{ color: 'var(--text-muted)' }}>React (Vite) + FastAPI</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>

                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 2: Text Extraction</strong>
                    <span style={{ color: 'var(--text-muted)' }}>PyMuPDF in-memory parsing</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>

                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 3: Semantic Search</strong>
                    <span style={{ color: 'var(--text-muted)' }}>Chunking + MiniLM + FAISS</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>

                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 4: RAG Pipeline</strong>
                    <span style={{ color: 'var(--text-muted)' }}>OpenAI Responses API + Grounded Q&A</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>

                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 5: Document Intelligence</strong>
                    <span style={{ color: 'var(--text-muted)' }}>Structured Summaries & Paper Comparison</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>

                <div className="arch-item">
                  <div className="arch-item-left">
                    <strong>Phase 6: Production Readiness</strong>
                    <span style={{ color: 'var(--text-muted)' }}>Security, Caching, Tests & Accessibility</span>
                  </div>
                  <span className="arch-tag arch-tag-active">Completed</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      <footer className="footer-note">
        DocuMind • Intelligent Document Research Assistant • Phase 6 Production Ready
      </footer>
    </div>
  );
}

