import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import HealthCard from './components/HealthCard';
import DocumentUpload from './components/DocumentUpload';
import DocumentViewer from './components/DocumentViewer';
import CompareDocuments from './components/CompareDocuments';
import { checkBackendHealth, listUploadedDocuments } from './services/api';
import './App.css';

const VALID_SCREENS = ['dashboard', 'documents', 'research', 'compare', 'settings'];

function getScreenFromHash() {
  const hash = (window.location.hash || '').replace('#', '').toLowerCase();
  return VALID_SCREENS.includes(hash) ? hash : 'dashboard';
}

export default function App() {
  const [activeScreen, setActiveScreen] = useState(getScreenFromHash);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [llmProvider, setLlmProvider] = useState('mock');
  const [registeredDocs, setRegisteredDocs] = useState([]);
  const [extractedDocument, setExtractedDocument] = useState(null);

  // Sync state with URL hash for browser Back/Forward navigation
  useEffect(() => {
    const handleHashChange = () => {
      const screen = getScreenFromHash();
      setActiveScreen(screen);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigateTo = useCallback((screenId) => {
    if (VALID_SCREENS.includes(screenId)) {
      window.location.hash = `#${screenId}`;
      setActiveScreen(screenId);
    }
  }, []);

  // Fetch registered documents from backend session
  const refreshDocuments = useCallback(async () => {
    try {
      const list = await listUploadedDocuments();
      setRegisteredDocs(list || []);
      // If we don't have an extractedDocument object but documents exist in session, keep track
    } catch {
      // Backend may be offline or empty
    }
  }, []);

  // Initial health check and documents list
  useEffect(() => {
    checkBackendHealth()
      .then((res) => {
        setBackendStatus('online');
        if (res?.data?.llm_provider) {
          setLlmProvider(res.data.llm_provider);
        }
      })
      .catch(() => setBackendStatus('offline'));

    refreshDocuments();
  }, [refreshDocuments]);

  const handleExtractionSuccess = (docData) => {
    setExtractedDocument(docData);
    refreshDocuments();
    // Navigate smoothly to the Research AI workspace for the uploaded document
    navigateTo('research');
  };

  const handleDocumentDeleted = () => {
    setExtractedDocument(null);
    refreshDocuments();
  };

  const indexedCount = registeredDocs.filter((d) => d.is_indexed).length;

  return (
    <div className="app-shell">
      {/* 1. Persistent Responsive Sidebar */}
      <Sidebar
        activeScreen={activeScreen}
        onNavigate={navigateTo}
        documentCount={registeredDocs.length}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        provider={llmProvider}
      />

      {/* 2. Main Application Wrapper */}
      <div className="app-main-wrapper">
        {/* TopBar with page title, active doc, connection & provider indicators */}
        <TopBar
          activeScreen={activeScreen}
          backendStatus={backendStatus}
          provider={llmProvider}
          activeDocument={extractedDocument}
          onOpenMobile={() => setMobileSidebarOpen(true)}
          onNavigate={navigateTo}
        />

        {/* 3. Main Workspace Content Area */}
        <main className="app-main-content">
          {/* ================= SCREEN: DASHBOARD ================= */}
          {activeScreen === 'dashboard' && (
            <div className="screen-container">
              {/* Real Metrics Widget Bar */}
              <div className="dashboard-overview-grid">
                <div className="stat-widget-card">
                  <div className="stat-widget-icon">📁</div>
                  <div className="stat-widget-content">
                    <span className="stat-widget-label">Active Documents</span>
                    <span className="stat-widget-value">{registeredDocs.length}</span>
                  </div>
                </div>

                <div className="stat-widget-card">
                  <div className="stat-widget-icon">⚡</div>
                  <div className="stat-widget-content">
                    <span className="stat-widget-label">FAISS Vector Indices</span>
                    <span className="stat-widget-value">{indexedCount}</span>
                  </div>
                </div>

                <div className="stat-widget-card">
                  <div className="stat-widget-icon">🧪</div>
                  <div className="stat-widget-content">
                    <span className="stat-widget-label">LLM Provider</span>
                    <span className="stat-widget-value" style={{ textTransform: 'capitalize' }}>
                      {llmProvider === 'openai' ? 'OpenAI GPT' : 'Mock (Local)'}
                    </span>
                  </div>
                </div>

                <div className="stat-widget-card">
                  <div className="stat-widget-icon">🌐</div>
                  <div className="stat-widget-content">
                    <span className="stat-widget-label">Backend Status</span>
                    <span
                      className="stat-widget-value"
                      style={{
                        color:
                          backendStatus === 'online'
                            ? '#6ee7b7'
                            : backendStatus === 'checking'
                            ? '#fcd34d'
                            : '#fda4af',
                        textTransform: 'capitalize',
                      }}
                    >
                      {backendStatus}
                    </span>
                  </div>
                </div>
              </div>

              {/* Active Document Spotlight Card if loaded */}
              {extractedDocument && (
                <div className="screen-header-card">
                  <div className="screen-header-text">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                      <span className="badge badge-health-online">Active In Workspace</span>
                      <span className="badge">{extractedDocument.total_pages} Pages</span>
                    </div>
                    <h2>{extractedDocument.filename}</h2>
                    <p>Loaded and ready for grounded RAG research, citations, and summaries.</p>
                  </div>
                  <div className="quick-actions-row">
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => navigateTo('research')}
                    >
                      Open in Research AI →
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => navigateTo('compare')}
                    >
                      Compare This Document
                    </button>
                  </div>
                </div>
              )}

              {/* Primary Document Upload Area */}
              <DocumentUpload
                onExtractionSuccess={handleExtractionSuccess}
                isBackendOnline={backendStatus}
              />

              {/* Registered Documents in Current Session */}
              {registeredDocs.length > 0 && (
                <div className="registered-docs-card">
                  <div className="registered-docs-header">
                    <div className="registered-docs-title">
                      <span>📑</span> Documents in In-Memory Session ({registeredDocs.length})
                    </div>
                    <button
                      type="button"
                      className="btn-secondary"
                      style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                      onClick={() => navigateTo('documents')}
                    >
                      View Full Library →
                    </button>
                  </div>

                  {registeredDocs.map((doc) => (
                    <div
                      key={doc.document_id}
                      className={`registered-doc-item ${
                        extractedDocument?.document_id === doc.document_id ? 'selected' : ''
                      }`}
                    >
                      <div className="doc-item-left">
                        <span style={{ fontSize: '1.25rem' }}>📄</span>
                        <div>
                          <div className="doc-item-title">{doc.filename}</div>
                          <div className="doc-item-meta">
                            {doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'} •{' '}
                            {doc.is_indexed ? `Indexed (${doc.chunk_count} chunks)` : 'Not Indexed'}
                          </div>
                        </div>
                      </div>
                      <div className="doc-item-actions">
                        <button
                          type="button"
                          className="btn-secondary"
                          style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                          onClick={() => {
                            if (extractedDocument?.document_id === doc.document_id) {
                              navigateTo('research');
                            } else {
                              // If clicked, we direct to research view
                              navigateTo('research');
                            }
                          }}
                        >
                          Research AI
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ================= SCREEN: DOCUMENTS ================= */}
          {activeScreen === 'documents' && (
            <div className="screen-container">
              <div className="screen-header-card">
                <div className="screen-header-text">
                  <h2>Document Library & Ingestion</h2>
                  <p>Upload PDFs to parse text into in-memory session structures and generate FAISS vector indices.</p>
                </div>
                <div className="quick-actions-row">
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={refreshDocuments}
                    title="Reload registered documents from session"
                  >
                    ↻ Refresh Library
                  </button>
                </div>
              </div>

              {/* Upload Dropzone */}
              <DocumentUpload
                onExtractionSuccess={handleExtractionSuccess}
                isBackendOnline={backendStatus}
              />

              {/* Session Documents List */}
              <div className="registered-docs-card">
                <div className="registered-docs-header">
                  <div className="registered-docs-title">
                    <span>📚</span> Registered Session Documents ({registeredDocs.length})
                  </div>
                  <span className="badge">In-Memory Storage</span>
                </div>

                {registeredDocs.length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', margin: '0.5rem 0' }}>
                    No documents are currently registered in this backend session. Drag and drop a PDF above to begin.
                  </p>
                ) : (
                  registeredDocs.map((doc) => (
                    <div
                      key={doc.document_id}
                      className={`registered-doc-item ${
                        extractedDocument?.document_id === doc.document_id ? 'selected' : ''
                      }`}
                    >
                      <div className="doc-item-left">
                        <span style={{ fontSize: '1.25rem' }}>📄</span>
                        <div>
                          <div className="doc-item-title">{doc.filename}</div>
                          <div className="doc-item-meta">
                            {doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'} •{' '}
                            {doc.is_indexed ? `Indexed (${doc.chunk_count} chunks)` : 'Not Indexed'}
                          </div>
                        </div>
                      </div>

                      <div className="doc-item-actions">
                        <button
                          type="button"
                          className="btn-primary"
                          style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                          onClick={() => {
                            if (extractedDocument?.document_id === doc.document_id) {
                              navigateTo('research');
                            } else {
                              navigateTo('research');
                            }
                          }}
                        >
                          Open in Research AI
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* ================= SCREEN: RESEARCH AI ================= */}
          {activeScreen === 'research' && (
            <div className="screen-container">
              {extractedDocument ? (
                <DocumentViewer
                  documentData={extractedDocument}
                  onReset={handleDocumentDeleted}
                />
              ) : (
                <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🧠</div>
                  <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>No Active Document Selected</h3>
                  <p style={{ color: 'var(--text-muted)', maxWidth: 500, margin: '0 auto 1.5rem', fontSize: '0.9rem' }}>
                    Select an existing document from your session library or upload a new PDF to activate the
                    grounded RAG question-answering workspace and structured summary generator.
                  </p>

                  <div style={{ maxWidth: 600, margin: '0 auto', textAlign: 'left' }}>
                    <DocumentUpload
                      onExtractionSuccess={handleExtractionSuccess}
                      isBackendOnline={backendStatus}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= SCREEN: COMPARE PAPERS ================= */}
          {activeScreen === 'compare' && (
            <div className="screen-container">
              <CompareDocuments
                currentDocumentId={extractedDocument?.document_id}
                onClose={() => navigateTo('dashboard')}
              />
            </div>
          )}

          {/* ================= SCREEN: SETTINGS & DIAGNOSTICS ================= */}
          {activeScreen === 'settings' && (
            <div className="screen-container">
              <div className="dashboard-grid">
                <HealthCard onHealthUpdate={setBackendStatus} />

                <div className="card">
                  <div className="card-title">
                    <span>System Runtime & Architecture</span>
                  </div>
                  <p className="card-subtitle">
                    Backend configuration, in-memory lifecycle policies, and active model providers.
                  </p>

                  <div className="arch-list">
                    <div className="arch-item">
                      <div className="arch-item-left">
                        <strong>Active LLM Provider</strong>
                        <span style={{ color: 'var(--text-muted)' }}>
                          {llmProvider === 'openai'
                            ? 'OpenAI API (GPT model with structured JSON responses)'
                            : 'Deterministic Mock Mode (FAISS vector retrieval, zero API costs)'}
                        </span>
                      </div>
                      <span className={`arch-tag ${llmProvider === 'openai' ? 'arch-tag-active' : ''}`}>
                        {llmProvider.toUpperCase()}
                      </span>
                    </div>

                    <div className="arch-item">
                      <div className="arch-item-left">
                        <strong>Storage Lifecycle</strong>
                        <span style={{ color: 'var(--text-muted)' }}>
                          FastAPI in-memory registry. Documents and vector indices exist for the server process lifecycle.
                        </span>
                      </div>
                      <span className="arch-tag arch-tag-active">In-Memory</span>
                    </div>

                    <div className="arch-item">
                      <div className="arch-item-left">
                        <strong>Embedding Model</strong>
                        <span style={{ color: 'var(--text-muted)' }}>
                          Sentence-Transformers (all-MiniLM-L6-v2) + FAISS IndexFlatL2
                        </span>
                      </div>
                      <span className="arch-tag arch-tag-active">384-dim</span>
                    </div>

                    <div className="arch-item">
                      <div className="arch-item-left">
                        <strong>Navigation Architecture</strong>
                        <span style={{ color: 'var(--text-muted)' }}>
                          Lightweight URL hash routing (#dashboard, #documents, #research, #compare, #settings)
                        </span>
                      </div>
                      <span className="arch-tag arch-tag-active">Native Hash</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>

        <footer className="shell-footer-note">
          DocuMind • Intelligent Document Research Workspace • Phase 3 Shell Completed
        </footer>
      </div>
    </div>
  );
}
