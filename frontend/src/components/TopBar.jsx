import React from 'react';

const SCREEN_METADATA = {
  dashboard: {
    title: 'Workspace Dashboard',
    subtitle: 'Overview of active research documents, upload pipelines, and intelligence tools.',
    icon: '📊',
  },
  documents: {
    title: 'Document Library',
    subtitle: 'Inspect, manage, and index uploaded PDF documents in your active session.',
    icon: '📁',
  },
  research: {
    title: 'Research AI Workspace',
    subtitle: 'Ask grounded questions, verify page citations, and generate structured summaries.',
    icon: '🧠',
  },
  compare: {
    title: 'Research Paper Comparison',
    subtitle: 'Perform side-by-side comparative analysis of scopes, methods, findings, and gaps.',
    icon: '⚖️',
  },
  settings: {
    title: 'Settings & Diagnostics',
    subtitle: 'Verify backend connectivity, test endpoint latency, and inspect runtime configurations.',
    icon: '⚙️',
  },
};

export default function TopBar({
  activeScreen = 'dashboard',
  backendStatus = 'checking',
  provider = 'mock',
  activeDocument = null,
  onOpenMobile,
  onNavigate,
}) {
  const currentMeta = SCREEN_METADATA[activeScreen] || SCREEN_METADATA.dashboard;

  const renderConnectionBadge = () => {
    if (backendStatus === 'online') {
      return (
        <div className="status-badge status-badge-online" title="Backend API is connected and responding">
          <span className="status-dot status-dot-online" aria-hidden="true"></span>
          <span>Backend Connected</span>
        </div>
      );
    }
    if (backendStatus === 'checking') {
      return (
        <div className="status-badge status-badge-checking" title="Checking backend health...">
          <span className="status-dot status-dot-checking" aria-hidden="true"></span>
          <span>Checking API...</span>
        </div>
      );
    }
    return (
      <div className="status-badge status-badge-offline" title="Backend is offline. Ensure FastAPI is running on port 8000.">
        <span className="status-dot status-dot-offline" aria-hidden="true"></span>
        <span>Backend Offline</span>
      </div>
    );
  };

  const renderProviderBadge = () => {
    if (provider === 'openai') {
      return (
        <div className="provider-badge provider-badge-openai" title="OpenAI API active">
          <span>⚡ OpenAI Active</span>
        </div>
      );
    }
    if (provider === 'mock') {
      return (
        <div className="provider-badge provider-badge-mock" title="Deterministic Mock mode with real FAISS retrieval">
          <span>🧪 Mock Mode</span>
        </div>
      );
    }
    return (
      <div className="provider-badge provider-badge-unknown" title="LLM Provider Unknown">
        <span>Provider Pending</span>
      </div>
    );
  };

  return (
    <header className="app-topbar">
      <div className="topbar-left">
        {/* Mobile menu button */}
        <button
          type="button"
          className="topbar-mobile-toggle"
          onClick={onOpenMobile}
          aria-label="Open navigation sidebar"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
          </svg>
        </button>

        <div className="topbar-title-wrap">
          <div className="topbar-breadcrumb">
            <span className="breadcrumb-root">DocuMind</span>
            <span className="breadcrumb-separator">/</span>
            <span className="breadcrumb-current">{currentMeta.title}</span>
          </div>
          <p className="topbar-subtitle">{currentMeta.subtitle}</p>
        </div>
      </div>

      <div className="topbar-right">
        {/* Active Document Indicator Chip */}
        {activeDocument && (
          <button
            type="button"
            className="topbar-active-doc-chip"
            onClick={() => onNavigate && onNavigate('research')}
            title={`Active Document: ${activeDocument.filename}. Click to open Research AI.`}
          >
            <span className="doc-chip-icon">📄</span>
            <span className="doc-chip-name">{activeDocument.filename}</span>
            {activeDocument.is_indexed && <span className="doc-chip-indexed">Indexed</span>}
          </button>
        )}

        {/* Real Status Indicators */}
        <div className="topbar-status-group">
          {renderProviderBadge()}
          {renderConnectionBadge()}
        </div>
      </div>
    </header>
  );
}
