import React from 'react';

const NAV_ITEMS = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    hash: '#dashboard',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="9"></rect>
        <rect x="14" y="3" width="7" height="5"></rect>
        <rect x="14" y="12" width="7" height="9"></rect>
        <rect x="3" y="16" width="7" height="5"></rect>
      </svg>
    ),
  },
  {
    id: 'documents',
    label: 'Documents',
    hash: '#documents',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
        <polyline points="10 9 9 9 8 9"></polyline>
      </svg>
    ),
  },
  {
    id: 'research',
    label: 'Research AI',
    hash: '#research',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
      </svg>
    ),
  },
  {
    id: 'compare',
    label: 'Compare Papers',
    hash: '#compare',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"></line>
        <line x1="12" y1="20" x2="12" y2="4"></line>
        <line x1="6" y1="20" x2="6" y2="14"></line>
      </svg>
    ),
  },
  {
    id: 'settings',
    label: 'Settings & Diagnostics',
    hash: '#settings',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
    ),
  },
];

export default function Sidebar({
  activeScreen,
  onNavigate,
  documentCount = 0,
  mobileOpen = false,
  onCloseMobile,
  provider = 'mock',
}) {
  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside className={`app-sidebar ${mobileOpen ? 'sidebar-mobile-open' : ''}`}>
        {/* Brand Header */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
          </div>
          <div className="sidebar-brand-text">
            <h1 className="sidebar-brand-title">DocuMind</h1>
            <span className="sidebar-brand-badge">Research Workspace</span>
          </div>

          {/* Close button on mobile screens */}
          {mobileOpen && (
            <button
              type="button"
              className="sidebar-close-btn"
              onClick={onCloseMobile}
              aria-label="Close navigation sidebar"
            >
              ✕
            </button>
          )}
        </div>

        {/* Navigation Links */}
        <nav className="sidebar-nav" aria-label="Main Navigation">
          <div className="sidebar-nav-group-title">WORKSPACE</div>
          <ul className="sidebar-nav-list">
            {NAV_ITEMS.map((item) => {
              const isActive = activeScreen === item.id;
              return (
                <li key={item.id} className="sidebar-nav-item">
                  <a
                    href={item.hash}
                    className={`sidebar-nav-link ${isActive ? 'active' : ''}`}
                    onClick={(e) => {
                      e.preventDefault();
                      onNavigate(item.id);
                      if (onCloseMobile) onCloseMobile();
                    }}
                    aria-current={isActive ? 'page' : undefined}
                  >
                    <span className="sidebar-nav-icon" aria-hidden="true">
                      {item.icon}
                    </span>
                    <span className="sidebar-nav-label">{item.label}</span>
                    {item.id === 'documents' && documentCount > 0 && (
                      <span className="sidebar-count-badge" title={`${documentCount} registered documents`}>
                        {documentCount}
                      </span>
                    )}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Sidebar Footer Info */}
        <div className="sidebar-footer">
          <div className="sidebar-status-box">
            <div className="sidebar-status-header">
              <span className="status-label-small">AI Engine</span>
              <span className={`provider-pill ${provider === 'openai' ? 'provider-openai' : 'provider-mock'}`}>
                {provider === 'openai' ? '⚡ OpenAI' : '🧪 Mock Mode'}
              </span>
            </div>
            <p className="sidebar-status-desc">
              {provider === 'openai'
                ? 'Connected to OpenAI API for synthesis & embeddings.'
                : 'Zero OpenAI credits used. In-memory deterministic mode.'}
            </p>
          </div>

          <div className="sidebar-footer-note">
            <span>In-Memory Session</span>
            <span className="status-dot-active" aria-hidden="true"></span>
          </div>
        </div>
      </aside>
    </>
  );
}
