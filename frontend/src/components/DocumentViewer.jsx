import React, { useState } from 'react';
import { askDocumentQuestion, deleteDocument, indexDocument, searchDocument, summarizeDocument } from '../services/api';
import CompareDocuments from './CompareDocuments';

export default function DocumentViewer({ documentData, onReset }) {
  const [selectedPageIndex, setSelectedPageIndex] = useState('all');
  const [keywordQuery, setKeywordQuery] = useState('');
  const [copiedId, setCopiedId] = useState(false);
  const [copiedPage, setCopiedPage] = useState(null);
  const [copiedSnippet, setCopiedSnippet] = useState(null);

  // Phase 3 Semantic Search & Indexing States
  const [isIndexed, setIsIndexed] = useState(Boolean(documentData?.is_indexed));
  const [indexing, setIndexing] = useState(false);
  const [indexInfo, setIndexInfo] = useState(null);
  const [indexingError, setIndexingError] = useState(null);

  const [semanticQuery, setSemanticQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [searchError, setSearchError] = useState(null);

  // Phase 4 RAG Question Answering States
  const [ragQuestion, setRagQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const [ragAnswerData, setRagAnswerData] = useState(null);
  const [ragError, setRagError] = useState(null);

  // Phase 5 Summarization & Document Comparison States
  const [summaryData, setSummaryData] = useState(null);
  const [summarizing, setSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);

  // Phase 6 Resource Lifecycle States
  const [deletingDoc, setDeletingDoc] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  // Active view: 'rag_qa' | 'summary' | 'search_results' | 'pages'
  const [activeTab, setActiveTab] = useState('rag_qa');

  if (!documentData) return null;

  const { filename, document_id, total_pages, total_words, total_characters, pages } = documentData;

  const sampleQuestions = [
    'What is the core thesis or objective of this document?',
    'What methodology or approach is proposed?',
    'What are the primary findings, outcomes, or conclusions?',
  ];

  const handleCopyDocId = () => {
    navigator.clipboard.writeText(document_id);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

  const handleCopyPageText = (pageNum, text) => {
    navigator.clipboard.writeText(text);
    setCopiedPage(pageNum);
    setTimeout(() => setCopiedPage(null), 2000);
  };

  const handleCopySnippet = (chunkId, text) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(chunkId);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  const handleIndex = async () => {
    if (indexing) return;
    setIndexing(true);
    setIndexingError(null);
    try {
      const res = await indexDocument(document_id, { chunkSize: 500, chunkOverlap: 100 });
      setIsIndexed(true);
      setIndexInfo(res);
    } catch (err) {
      setIndexingError(err.message || 'Failed to index document.');
    } finally {
      setIndexing(false);
    }
  };

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (searching || !semanticQuery.trim()) return;

    setSearching(true);
    setSearchError(null);
    try {
      const res = await searchDocument(document_id, semanticQuery, 5);
      setSearchResults(res.results || []);
      setActiveTab('search_results');
    } catch (err) {
      setSearchError(err.message || 'Semantic search failed.');
    } finally {
      setSearching(false);
    }
  };

  const handleAskQuestion = async (e, customQ = null) => {
    e?.preventDefault();
    if (asking) return;
    const queryToAsk = (customQ !== null ? customQ : ragQuestion).trim();
    if (!queryToAsk) return;

    if (customQ !== null) {
      setRagQuestion(customQ);
    }

    setAsking(true);
    setRagError(null);
    try {
      const res = await askDocumentQuestion(document_id, queryToAsk, 4);
      setRagAnswerData(res);
      setActiveTab('rag_qa');
    } catch (err) {
      setRagError(err.message || 'Failed to generate answer.');
    } finally {
      setAsking(false);
    }
  };

  const handleSummarize = async () => {
    if (summarizing) return;
    setSummarizing(true);
    setSummaryError(null);
    try {
      const res = await summarizeDocument(document_id);
      setSummaryData(res);
      setActiveTab('summary');
    } catch (err) {
      setSummaryError(err.message || 'Failed to generate document summary.');
    } finally {
      setSummarizing(false);
    }
  };

  const handleDeleteDoc = async () => {
    if (deletingDoc) return;
    if (!window.confirm(`Permanently remove "${filename}" from session memory? All extracted text and FAISS vector indices will be deleted.`)) {
      return;
    }
    setDeletingDoc(true);
    setDeleteError(null);
    try {
      await deleteDocument(document_id);
      if (onReset) onReset();
    } catch (err) {
      setDeleteError(err.message || 'Failed to delete document from memory.');
      setDeletingDoc(false);
    }
  };

  const jumpToPage = (pageNum) => {
    setSelectedPageIndex(pageNum - 1);
    setActiveTab('pages');
  };

  // Filter pages for standard document view
  const filteredPages = pages.filter((p, idx) => {
    if (selectedPageIndex !== 'all' && selectedPageIndex !== idx) {
      return false;
    }
    if (keywordQuery.trim() !== '') {
      return p.text.toLowerCase().includes(keywordQuery.toLowerCase());
    }
    return true;
  });

  if (showComparison) {
    return (
      <CompareDocuments
        currentDocumentId={document_id}
        onClose={() => setShowComparison(false)}
      />
    );
  }

  return (
    <div className="card viewer-card">
      {/* 1. Document Overview Header */}
      <div className="viewer-header">
        <div className="viewer-title-row">
          <div>
            <div className="viewer-badge-row">
              <span className="badge badge-health-online">Extracted Successfully</span>
              <span className={`badge ${isIndexed ? 'badge-health-online' : 'badge-phase'}`}>
                {isIndexed ? '✓ Indexed (RAG Ready)' : 'Not Indexed'}
              </span>
              <span className="badge">{total_pages} {total_pages === 1 ? 'Page' : 'Pages'}</span>
              <span className="badge">{total_words.toLocaleString()} Words</span>
              <span className="badge">{total_characters.toLocaleString()} Chars</span>
            </div>
            <h2 className="viewer-filename">{filename}</h2>
            <div className="viewer-doc-id">
              <code>ID: {document_id}</code>
              <button
                className="btn-tiny-copy"
                onClick={handleCopyDocId}
                title="Copy Document ID"
              >
                {copiedId ? '✓ Copied' : 'Copy ID'}
              </button>
            </div>
          </div>

          <div className="header-action-buttons">
            <button
              className="btn-secondary"
              onClick={() => setShowComparison(true)}
              title="Compare this document with another research paper"
            >
              ⚖️ Compare Papers
            </button>
            <button
              className="btn-secondary"
              style={{ color: '#f87171', borderColor: 'rgba(239, 68, 68, 0.3)' }}
              onClick={handleDeleteDoc}
              disabled={deletingDoc}
              title="Permanently remove this document and its FAISS index from session memory"
            >
              {deletingDoc ? 'Deleting...' : '🗑️ Delete Document'}
            </button>
            <button className="btn-secondary" onClick={onReset}>
              Upload Another PDF
            </button>
          </div>
        </div>

        {deleteError && (
          <div className="alert-banner alert-danger" style={{ marginTop: '0.75rem' }}>
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Deletion Failed:</strong>
              <p>{deleteError}</p>
            </div>
          </div>
        )}

        {/* 2. Semantic Indexing Action Box */}
        <div className={`indexing-box ${isIndexed ? 'indexing-box-ready' : ''}`}>
          <div className="indexing-info">
            <div className="indexing-title">
              <span className="indexing-dot" style={{ backgroundColor: isIndexed ? '#10b981' : '#f59e0b' }}></span>
              <strong>Document Knowledge Index (FAISS + Sentence Transformers)</strong>
            </div>
            <p className="indexing-desc">
              {isIndexed
                ? `Active in memory. ${indexInfo?.total_chunks || 'Document'} chunks indexed with ${indexInfo?.embedding_model || 'all-MiniLM-L6-v2'} ready for RAG question answering.`
                : 'Build an in-memory vector index to enable AI Question Answering and semantic concept retrieval.'}
            </p>
          </div>

          <div className="indexing-actions">
            {!isIndexed ? (
              <button
                className="btn-primary"
                onClick={handleIndex}
                disabled={indexing}
              >
                {indexing ? (
                  <>
                    <span className="btn-spinner"></span> Indexing Document...
                  </>
                ) : (
                  'Index Document for RAG'
                )}
              </button>
            ) : (
              <button
                className="btn-secondary"
                onClick={handleIndex}
                disabled={indexing}
                title="Re-chunk and rebuild FAISS index"
              >
                {indexing ? 'Re-indexing...' : 'Re-index Document'}
              </button>
            )}
          </div>
        </div>

        {indexingError && (
          <div className="alert-banner alert-danger" style={{ marginTop: '0.75rem' }}>
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Indexing Error:</strong>
              <p>{indexingError}</p>
            </div>
          </div>
        )}

        {/* 3. Navigation View Tabs (RAG Q&A vs Semantic Search vs Full Pages) */}
        <div className="viewer-nav-bar">
          <div className="view-mode-tabs" role="tablist" aria-label="Document view modes">
            <button
              role="tab"
              aria-selected={activeTab === 'rag_qa'}
              className={`view-mode-tab ${activeTab === 'rag_qa' ? 'view-mode-tab-active' : ''}`}
              onClick={() => setActiveTab('rag_qa')}
            >
              Ask DocuMind (RAG)
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'summary'}
              className={`view-mode-tab ${activeTab === 'summary' ? 'view-mode-tab-active' : ''}`}
              onClick={() => {
                setActiveTab('summary');
                if (!summaryData && !summarizing) {
                  handleSummarize();
                }
              }}
            >
              Document Summary {summaryData ? '✓' : ''}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'search_results'}
              className={`view-mode-tab ${activeTab === 'search_results' ? 'view-mode-tab-active' : ''}`}
              onClick={() => setActiveTab('search_results')}
            >
              Semantic Search {searchResults ? `(${searchResults.length})` : ''}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'pages'}
              className={`view-mode-tab ${activeTab === 'pages' ? 'view-mode-tab-active' : ''}`}
              onClick={() => setActiveTab('pages')}
            >
              Document Pages ({total_pages})
            </button>
          </div>

          {activeTab === 'pages' && (
            <div className="search-box">
              <input
                type="text"
                placeholder="Filter pages by keyword..."
                value={keywordQuery}
                onChange={(e) => setKeywordQuery(e.target.value)}
                className="search-input"
                aria-label="Filter pages by keyword"
              />
              {keywordQuery && (
                <button
                  type="button"
                  className="search-clear-btn"
                  onClick={() => setKeywordQuery('')}
                  aria-label="Clear keyword filter"
                >
                  ✕
                </button>
              )}
            </div>
          )}
        </div>

        {/* Page selector pills if in Pages view */}
        {activeTab === 'pages' && (
          <div className="page-tabs-scroll">
            <button
              className={`page-tab ${selectedPageIndex === 'all' ? 'page-tab-active' : ''}`}
              onClick={() => setSelectedPageIndex('all')}
            >
              All Pages
            </button>
            {pages.map((p, idx) => (
              <button
                key={p.page}
                className={`page-tab ${selectedPageIndex === idx ? 'page-tab-active' : ''}`}
                onClick={() => setSelectedPageIndex(idx)}
              >
                Page {p.page}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* 4. TAB 1: RAG Question Answering View */}
      {activeTab === 'rag_qa' && (
        <div className="rag-container">
          <form className="rag-question-form" onSubmit={handleAskQuestion}>
            <div className="semantic-input-wrap">
              <svg className="semantic-search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
              <input
                type="text"
                className="semantic-search-input"
                placeholder={isIndexed ? "Ask any research question about this document..." : "Index document above to ask questions..."}
                value={ragQuestion}
                onChange={(e) => setRagQuestion(e.target.value)}
                disabled={!isIndexed || asking}
              />
              {ragQuestion && (
                <button
                  type="button"
                  className="semantic-clear-btn"
                  onClick={() => setRagQuestion('')}
                >
                  ✕
                </button>
              )}
            </div>
            <button
              type="submit"
              className="btn-primary semantic-submit-btn"
              disabled={!isIndexed || !ragQuestion.trim() || asking}
            >
              {asking ? (
                <>
                  <span className="btn-spinner"></span> Synthesizing...
                </>
              ) : (
                'Ask DocuMind'
              )}
            </button>
          </form>

          {/* Quick Prompt Suggestions */}
          {isIndexed && !ragAnswerData && !asking && (
            <div className="quick-prompts-row">
              <span className="quick-prompts-label">Suggested:</span>
              {sampleQuestions.map((sq, i) => (
                <button
                  key={i}
                  className="quick-prompt-pill"
                  onClick={(e) => handleAskQuestion(e, sq)}
                >
                  "{sq}"
                </button>
              ))}
            </div>
          )}

          {ragError && (
            <div className="alert-banner alert-danger" style={{ marginTop: '1rem' }}>
              <span className="alert-icon">⚠️</span>
              <div className="alert-content">
                <strong>Question Answering Error:</strong>
                <p>{ragError}</p>
                <small className="alert-tip">
                  Tip: Ensure your OPENAI_API_KEY is configured in backend/.env.
                </small>
              </div>
            </div>
          )}

          {/* Synthesized Answer Output Card */}
          {ragAnswerData && (
            <div className={`rag-answer-card ${ragAnswerData.llm_provider === 'mock' ? 'rag-answer-card-mock' : ''}`}>
              {/* Prominent Mock Mode Callout if in mock mode */}
              {(ragAnswerData.llm_provider === 'mock' || ragAnswerData.model_used?.includes('mock')) && (
                <div className="mock-mode-callout">
                  <div className="mock-mode-callout-header">
                    <span className="mock-mode-pill">🧪 MOCK MODE — simulated answer, not AI-generated</span>
                    <span className="mock-mode-badge-free">Zero OpenAI Credits Used</span>
                  </div>
                  <p className="mock-mode-callout-text">
                    This answer is deterministically synthesized from <strong>real FAISS semantic retrieval</strong> and document chunks for cost-free testing.
                  </p>
                </div>
              )}

              <div className="rag-answer-header">
                <div className="rag-answer-title-group">
                  <div className={`rag-ai-icon ${ragAnswerData.llm_provider === 'mock' ? 'rag-mock-icon' : ''}`}>
                    {ragAnswerData.llm_provider === 'mock' ? 'MOCK' : 'AI'}
                  </div>
                  <div>
                    <h3 className="rag-answer-heading">
                      {ragAnswerData.llm_provider === 'mock' ? 'DocuMind Simulated Answer' : 'DocuMind Answer'}
                    </h3>
                    <span className="rag-model-badge">
                      {ragAnswerData.llm_provider === 'mock'
                        ? 'Provider: Mock Deterministic Synthesizer (Offline)'
                        : `Powered by ${ragAnswerData.model_used}`}
                    </span>
                  </div>
                </div>

                <div className="rag-evidence-badge-group">
                  {ragAnswerData.llm_provider === 'mock' && (
                    <span className="badge badge-mock-tag">Mock Mode</span>
                  )}
                  {ragAnswerData.has_sufficient_evidence ? (
                    <span className="badge badge-health-online">
                      ✓ Supported by {ragAnswerData.sources.length} Excerpt{ragAnswerData.sources.length === 1 ? '' : 's'}
                    </span>
                  ) : (
                    <span className="badge badge-health-checking">
                      ⚠️ Insufficient Evidence in Document
                    </span>
                  )}
                </div>
              </div>

              <div className="rag-answer-body">
                <p className="rag-answer-text">{ragAnswerData.answer}</p>
              </div>

              {/* Verified Sources & Citations Section */}
              {ragAnswerData.sources && ragAnswerData.sources.length > 0 && (
                <div className="rag-sources-section">
                  <h4 className="rag-sources-heading">
                    Verified Citations & Grounding Excerpts (Real FAISS Retrieval)
                  </h4>
                  <div className="rag-sources-grid">
                    {ragAnswerData.sources.map((src) => {
                      const matchPct = Math.round(src.score * 100);
                      return (
                        <div key={src.chunk_id || src.citation_index} className="rag-source-item">
                          <div className="rag-source-header">
                            <div className="rag-source-header-left">
                              <span className="citation-marker">[{src.citation_index}]</span>
                              <button
                                className="result-page-link"
                                onClick={() => jumpToPage(src.page)}
                                title="Click to view full source page"
                              >
                                Source: {src.filename ? `${src.filename} (Page ${src.page})` : `Page ${src.page}`} →
                              </button>
                              <span className="match-score-badge">
                                {matchPct}% Relevance
                              </span>
                            </div>
                            <div className="rag-source-header-right">
                              <span className="chunk-id-tag"><code>{src.chunk_id}</code></span>
                              <button
                                className="btn-tiny-copy"
                                onClick={() => handleCopySnippet(src.chunk_id, src.text)}
                              >
                                {copiedSnippet === src.chunk_id ? '✓ Copied' : 'Copy'}
                              </button>
                            </div>
                          </div>
                          <p className="rag-source-text">{src.text}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 5. TAB 2: Semantic Search Results View */}
      {activeTab === 'search_results' && (
        <div className="results-container">
          {/* Natural Language Semantic Search Input */}
          <form className="semantic-search-form" onSubmit={handleSearch}>
            <div className="semantic-input-wrap">
              <svg className="semantic-search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              <input
                type="text"
                className="semantic-search-input"
                placeholder={isIndexed ? "Search by concept in FAISS (e.g. 'experimental methodology', 'accuracy metrics')..." : "Index document above to search..."}
                value={semanticQuery}
                onChange={(e) => setSemanticQuery(e.target.value)}
                disabled={!isIndexed || searching}
              />
              {semanticQuery && (
                <button
                  type="button"
                  className="semantic-clear-btn"
                  onClick={() => setSemanticQuery('')}
                >
                  ✕
                </button>
              )}
            </div>
            <button
              type="submit"
              className="btn-primary semantic-submit-btn"
              disabled={!isIndexed || !semanticQuery.trim() || searching}
            >
              {searching ? (
                <>
                  <span className="btn-spinner"></span> Searching...
                </>
              ) : (
                'Search FAISS'
              )}
            </button>
          </form>

          {searchError && (
            <div className="alert-banner alert-danger" style={{ marginTop: '0.75rem' }}>
              <span className="alert-icon">⚠️</span>
              <div className="alert-content">
                <strong>Search Error:</strong>
                <p>{searchError}</p>
              </div>
            </div>
          )}

          {searchResults && (
            <>
              <div className="results-header-info" style={{ marginTop: '1.25rem' }}>
                <span>
                  Top retrieved passages for: <em>"{semanticQuery}"</em>
                </span>
                <button className="btn-secondary" onClick={() => setActiveTab('pages')}>
                  Back to Pages
                </button>
              </div>

              {searchResults.length === 0 ? (
                <div className="empty-results-box">
                  <p>No matching passages found for query "{semanticQuery}"</p>
                </div>
              ) : (
                <div className="results-list">
                  {searchResults.map((item, index) => {
                    const matchPct = Math.round(item.score * 100);
                    return (
                      <div key={item.chunk_id || index} className="result-card">
                        <div className="result-card-header">
                          <div className="result-badges">
                            <span className="result-rank-pill">#{index + 1}</span>
                            <span className="match-score-badge" title={`Cosine similarity: ${item.score}`}>
                              {matchPct}% Match
                            </span>
                            <button
                              className="result-page-link"
                              onClick={() => jumpToPage(item.page)}
                              title="Click to view full source page"
                            >
                              Source: Page {item.page} →
                            </button>
                            <span className="result-metrics">{item.word_count} words</span>
                          </div>

                          <button
                            className="btn-tiny-copy"
                            onClick={() => handleCopySnippet(item.chunk_id, item.text)}
                          >
                            {copiedSnippet === item.chunk_id ? '✓ Copied' : 'Copy'}
                          </button>
                        </div>

                        <div className="result-card-body">
                          <p className="result-passage-text">{item.text}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* 6. TAB 3: Standard Pages Reader View */}
      {activeTab === 'pages' && (
        <div className="pages-list">
          {filteredPages.length === 0 ? (
            <div className="empty-results-box">
              <p>No text found matching "{keywordQuery}"</p>
              <button className="btn-secondary" onClick={() => setKeywordQuery('')}>
                Clear Filter
              </button>
            </div>
          ) : (
            filteredPages.map((pageItem) => (
              <div key={pageItem.page} className="page-card">
                <div className="page-card-header">
                  <div className="page-card-title">
                    <span className="page-number-pill">Page {pageItem.page}</span>
                    <span className="page-metrics">
                      {pageItem.word_count.toLocaleString()} words • {pageItem.char_count.toLocaleString()} chars
                    </span>
                  </div>
                  <button
                    className="btn-copy-page"
                    onClick={() => handleCopyPageText(pageItem.page, pageItem.text)}
                  >
                    {copiedPage === pageItem.page ? '✓ Copied Page' : 'Copy Page Text'}
                  </button>
                </div>

                <div className="page-card-body">
                  <pre className="page-extracted-text">
                    {pageItem.text || '// (Empty or whitespace page)'}
                  </pre>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* 7. TAB 4: Phase 5 Document Summarization View */}
      {activeTab === 'summary' && (
        <div className="summary-container">
          <div className="summary-action-banner">
            <div className="summary-action-text">
              <h3 className="summary-title">Structured Document Intelligence</h3>
              <p className="summary-subtitle">
                Executive overview, key objectives, highlights, findings with page references, and conclusion.
              </p>
            </div>
            <button
              className="btn-primary"
              onClick={handleSummarize}
              disabled={summarizing}
            >
              {summarizing ? (
                <>
                  <span className="btn-spinner"></span> Synthesizing Summary...
                </>
              ) : summaryData ? (
                'Regenerate Summary'
              ) : (
                'Generate Summary'
              )}
            </button>
          </div>

          {summarizing && (
            <div className="summary-loading-box">
              <div className="btn-spinner" style={{ width: 28, height: 28, borderWidth: 3 }}></div>
              <p className="summary-loading-text">
                Extracting core sections, objectives, and evidence findings from document...
              </p>
            </div>
          )}

          {summaryError && (
            <div className="alert-banner alert-danger" style={{ marginTop: '1rem' }}>
              <span className="alert-icon">⚠️</span>
              <div className="alert-content">
                <strong>Summarization Error:</strong>
                <p>{summaryError}</p>
              </div>
            </div>
          )}

          {!summarizing && summaryData && (
            <div className="summary-content-card">
              {summaryData.is_simulated ? (
                <div className="mock-mode-callout" style={{ borderRadius: 'var(--radius-md)', marginBottom: '1.25rem' }}>
                  <div className="mock-mode-callout-header">
                    <span className="mock-mode-pill">
                      🧪 MOCK MODE — simulated summary, not AI-generated
                    </span>
                    <span className="mock-mode-badge-free">Zero OpenAI Credits Used</span>
                  </div>
                  <p className="mock-mode-callout-text">
                    Synthesized deterministically from actual extracted text pages. Set <code>LLM_PROVIDER=openai</code> in <code>backend/.env</code> for live OpenAI generation.
                  </p>
                </div>
              ) : (
                <div className="mock-mode-callout" style={{ borderRadius: 'var(--radius-md)', marginBottom: '1.25rem', borderColor: 'rgba(59, 130, 246, 0.3)' }}>
                  <div className="mock-mode-callout-header">
                    <span className="mock-mode-pill" style={{ color: '#60a5fa' }}>
                      🤖 OpenAI Model: {summaryData.model_used}
                    </span>
                    <span className="badge badge-health-online">Grounded AI Analysis</span>
                  </div>
                </div>
              )}

              {/* Executive Overview */}
              <div className="summary-section">
                <h4 className="summary-section-heading">📄 Executive Overview</h4>
                <p className="summary-text-lead">{summaryData.overview}</p>
              </div>

              {/* Research Objectives */}
              {summaryData.objectives && summaryData.objectives.length > 0 && (
                <div className="summary-section">
                  <h4 className="summary-section-heading">🎯 Research Objectives</h4>
                  <div className="summary-objectives-grid">
                    {summaryData.objectives.map((obj, i) => (
                      <div key={i} className="summary-objective-item">
                        <span className="objective-number">{i + 1}</span>
                        <span className="objective-text">{obj}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Core Highlights */}
              {summaryData.key_points && summaryData.key_points.length > 0 && (
                <div className="summary-section">
                  <h4 className="summary-section-heading">💡 Core Highlights & Key Points</h4>
                  <ul className="summary-bullet-list">
                    {summaryData.key_points.map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Findings & Evidence */}
              {summaryData.findings && summaryData.findings.length > 0 && (
                <div className="summary-section">
                  <h4 className="summary-section-heading">📊 Key Findings & Evidence</h4>
                  <div className="findings-cards-grid">
                    {summaryData.findings.map((item, i) => {
                      const pageNum = item.page ?? item.page_number;
                      const findingText = item.text ?? item.finding;
                      return (
                        <div key={i} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-marker">Finding #{i + 1}</span>
                            <button
                              type="button"
                              className="result-page-link"
                              onClick={() => jumpToPage(pageNum)}
                              title={`Jump to source page ${pageNum}`}
                              aria-label={`Jump to source page ${pageNum}`}
                            >
                              Source: Page {pageNum} →
                            </button>
                          </div>
                          <p className="finding-text">{findingText}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Conclusion */}
              <div className="summary-section">
                <h4 className="summary-section-heading">🏁 Conclusion</h4>
                <p className="summary-text-conclusion">{summaryData.conclusion}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
