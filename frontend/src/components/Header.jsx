import React from 'react';

export default function Header({ backendStatus }) {
  const getStatusBadge = () => {
    if (backendStatus === 'online') {
      return (
        <span className="badge badge-health-online">
          <span className="pulse-dot"></span> Backend Connected
        </span>
      );
    }
    if (backendStatus === 'checking') {
      return (
        <span className="badge badge-health-checking">
          <span className="pulse-dot"></span> Checking Backend...
        </span>
      );
    }
    return (
      <span className="badge badge-health-offline">
        <span className="pulse-dot"></span> Backend Offline
      </span>
    );
  };

  return (
    <header className="header-wrapper">
      <div className="brand-section">
        <div className="brand-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
        </div>
        <div>
          <h1 className="brand-title">DocuMind</h1>
          <p className="brand-subtitle">Intelligent Document Research Assistant</p>
        </div>
      </div>

      <div className="header-badges">
        <span className="badge badge-phase">Phase 6: Production Ready</span>
        {getStatusBadge()}
      </div>
    </header>
  );
}
