import React, { useState, useEffect } from 'react';
import { listUploadedDocuments, compareDocuments } from '../services/api';

export default function CompareDocuments({ currentDocumentId, onClose }) {
  const [documents, setDocuments] = useState([]);
  const [docA, setDocA] = useState(currentDocumentId || '');
  const [docB, setDocB] = useState('');
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadingDocs, setLoadingDocs] = useState(true);

  useEffect(() => {
    loadDocs();
  }, []);

  const loadDocs = async () => {
    setLoadingDocs(true);
    try {
      const list = await listUploadedDocuments();
      setDocuments(list || []);
      if (list && list.length >= 2) {
        if (!docA && list[0]) setDocA(list[0].document_id);
        const secondDoc = list.find((d) => d.document_id !== (currentDocumentId || list[0]?.document_id));
        if (secondDoc) setDocB(secondDoc.document_id);
      }
    } catch (err) {
      setError('Failed to load active documents for comparison.');
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleCompare = async (e) => {
    e?.preventDefault();
    if (comparing) return;
    if (!docA || !docB) {
      setError('Please select two distinct documents.');
      return;
    }
    if (docA === docB) {
      setError('Cannot compare a document with itself. Please pick two different documents.');
      return;
    }

    setComparing(true);
    setError(null);
    const startTime = performance.now();
    try {
      const res = await compareDocuments(docA, docB);
      res.latency = Math.round(performance.now() - startTime);
      setComparisonResult(res);
    } catch (err) {
      setError(err.message || 'Comparison failed.');
    } finally {
      setComparing(false);
    }
  };

  return (
    <div className="card comparison-card">
      <div className="comparison-header">
        <div className="comparison-title-group">
          <div className="brand-icon" style={{ width: 40, height: 40, fontSize: '1.2rem' }}>⚖️</div>
          <div>
            <h3 className="viewer-filename" style={{ fontSize: '1.3rem' }}>Research Paper & Document Comparison</h3>
            <p className="card-subtitle" style={{ margin: 0 }}>
              Compare objectives, methodologies, findings, limitations, and evidence gaps between two uploaded documents.
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={loadDocs}
            disabled={loadingDocs || comparing}
            title="Reload active session documents"
            aria-label="Refresh active documents list"
          >
            {loadingDocs ? '↻ Refreshing...' : '↻ Refresh List'}
          </button>
          {onClose && (
            <button type="button" className="btn-secondary" onClick={onClose} aria-label="Back to document workspace">
              Back to Document
            </button>
          )}
        </div>
      </div>


      {/* Document Selection Row */}
      <form onSubmit={handleCompare} className="comparison-selector-form">
        <div className="comparison-selectors-row">
          <div className="selector-group">
            <label className="selector-label" htmlFor="compare-doc-a-select">Document A:</label>
            <select
              id="compare-doc-a-select"
              className="select-input"
              value={docA}
              onChange={(e) => setDocA(e.target.value)}
              disabled={comparing || loadingDocs}
              aria-label="Select Document A for comparison"
            >
              <option value="">Select Document A...</option>
              {documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.filename} ({d.total_pages} {d.total_pages === 1 ? 'page' : 'pages'})
                </option>
              ))}
            </select>
          </div>

          <div className="selector-vs-badge" aria-hidden="true">VS</div>

          <div className="selector-group">
            <label className="selector-label" htmlFor="compare-doc-b-select">Document B:</label>
            <select
              id="compare-doc-b-select"
              className="select-input"
              value={docB}
              onChange={(e) => setDocB(e.target.value)}
              disabled={comparing || loadingDocs}
              aria-label="Select Document B for comparison"
            >
              <option value="">Select Document B...</option>
              {documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.filename} ({d.total_pages} {d.total_pages === 1 ? 'page' : 'pages'})
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="btn-primary compare-submit-btn"
            disabled={comparing || !docA || !docB || docA === docB}
            aria-label={comparing ? "Comparing documents" : "Run comparison"}
          >
            {comparing ? (
              <>
                <span className="btn-spinner"></span> Comparing...
              </>
            ) : (
              'Compare Documents'
            )}
          </button>
        </div>
      </form>

      {documents.length < 2 && !loadingDocs && (
        <div className="alert-banner alert-info" style={{ marginTop: '1rem' }}>
          <span className="alert-icon">ℹ️</span>
          <div className="alert-content">
            <strong>Only 1 document is registered in memory.</strong>
            <p>To compare research papers, upload a second PDF using "Upload Another PDF" above. Both documents will stay in memory for comparison.</p>
          </div>
        </div>
      )}

      {error && (
        <div className="alert-banner alert-danger" style={{ marginTop: '1rem' }}>
          <span className="alert-icon">⚠️</span>
          <div className="alert-content">
            <strong>Comparison Error:</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Comparison Results Card */}
      {comparisonResult && (
        <div className="comparison-results-wrap">
          {/* Mock Mode Banner if simulated */}
          {(comparisonResult.is_simulated || comparisonResult.llm_provider === 'mock') && (
            <div className="mock-mode-callout" style={{ borderRadius: 'var(--radius-md)', marginBottom: '1.25rem' }}>
              <div className="mock-mode-callout-header">
                <span className="mock-mode-pill">🧪 MOCK MODE — simulated comparison, not AI-generated</span>
                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  {comparisonResult.latency && (
                    <span className="badge badge-health-online">✓ {comparisonResult.latency}ms</span>
                  )}
                  <span className="mock-mode-badge-free">Zero OpenAI Credits Used</span>
                </div>
              </div>

              <p className="mock-mode-callout-text">
                This comparative analysis was deterministically derived from the <strong>actual extracted text</strong> of both documents.
              </p>
            </div>
          )}

          {/* Side-by-side Profiles */}
          <div className="comparison-grid-two-col">
            <div className="comparison-doc-card">
              <div className="doc-card-badge-header">
                <span className="badge badge-phase">Document A</span>
                <span className="badge">{comparisonResult.document_a.total_pages} Pages</span>
              </div>
              <h4 className="comparison-doc-title">{comparisonResult.document_a.filename}</h4>

              <div className="facet-block">
                <strong className="facet-label">🎯 Objectives:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_a.objectives.map((obj, i) => (
                    <li key={i}>{obj}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">🔬 Methodologies:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_a.methodologies.map((m, i) => (
                    <li key={i}>{m}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">📊 Key Findings:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_a.findings.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">⚠️ Limitations:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_a.limitations.map((l, i) => (
                    <li key={i}>{l}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="comparison-doc-card">
              <div className="doc-card-badge-header">
                <span className="badge badge-phase">Document B</span>
                <span className="badge">{comparisonResult.document_b.total_pages} Pages</span>
              </div>
              <h4 className="comparison-doc-title">{comparisonResult.document_b.filename}</h4>

              <div className="facet-block">
                <strong className="facet-label">🎯 Objectives:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_b.objectives.map((obj, i) => (
                    <li key={i}>{obj}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">🔬 Methodologies:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_b.methodologies.map((m, i) => (
                    <li key={i}>{m}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">📊 Key Findings:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_b.findings.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>

              <div className="facet-block">
                <strong className="facet-label">⚠️ Limitations:</strong>
                <ul className="facet-list">
                  {comparisonResult.document_b.limitations.map((l, i) => (
                    <li key={i}>{l}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Similarities & Differences */}
          <div className="comparison-insights-grid">
            <div className="insight-card insight-similarities">
              <h4 className="insight-heading">🔗 Key Similarities</h4>
              <ul className="insight-list">
                {comparisonResult.similarities.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>

            <div className="insight-card insight-differences">
              <h4 className="insight-heading">⚡ Key Differences</h4>
              <ul className="insight-list">
                {comparisonResult.differences.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Missing Information / Evidence Gaps */}
          {comparisonResult.missing_information && comparisonResult.missing_information.length > 0 && (
            <div className="insight-card insight-missing">
              <h4 className="insight-heading">🔍 Missing Information & Evidence Gaps</h4>
              <ul className="insight-list">
                {comparisonResult.missing_information.map((gap, i) => (
                  <li key={i}>{gap}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Citations */}
          {comparisonResult.citations && comparisonResult.citations.length > 0 && (
            <div className="rag-sources-section" style={{ marginTop: '1.5rem', borderRadius: 'var(--radius-md)' }}>
              <h4 className="rag-sources-heading">Verified Document Citations</h4>
              <div className="rag-sources-grid">
                {comparisonResult.citations.map((c, i) => (
                  <div key={i} className="rag-source-item">
                    <div className="rag-source-header">
                      <div className="rag-source-header-left">
                        <span className="citation-marker">[{i + 1}]</span>
                        <span className="match-score-badge">{c.filename}</span>
                        <span className="result-page-link">Page {c.page}</span>
                      </div>
                    </div>
                    <p className="rag-source-text">{c.excerpt}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
