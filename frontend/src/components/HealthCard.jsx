import React, { useState } from 'react';
import { checkBackendHealth, fetchRootInfo } from '../services/api';

export default function HealthCard({ onHealthUpdate }) {
  const [loading, setLoading] = useState(false);
  const [activeEndpoint, setActiveEndpoint] = useState('/health');
  const [responseOutput, setResponseOutput] = useState(null);
  const [latency, setLatency] = useState(null);
  const [lastCheckTime, setLastCheckTime] = useState(null);

  const handleTestEndpoint = async (endpointType) => {
    setLoading(true);
    setActiveEndpoint(endpointType);
    try {
      let result;
      if (endpointType === '/health') {
        result = await checkBackendHealth();
        if (onHealthUpdate) onHealthUpdate('online');
      } else {
        result = await fetchRootInfo();
      }
      setResponseOutput(JSON.stringify(result.data, null, 2));
      setLatency(result.latency);
      setLastCheckTime(new Date().toLocaleTimeString());
    } catch (err) {
      setResponseOutput(
        JSON.stringify(
          {
            error: 'Connection Failed',
            details: err.message,
            hint: 'Ensure FastAPI backend is running via `uvicorn app.main:app --reload` on port 8000',
          },
          null,
          2
        )
      );
      setLatency(null);
      setLastCheckTime(new Date().toLocaleTimeString());
      if (onHealthUpdate) onHealthUpdate('offline');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>Backend Verification</span>
        {latency !== null && (
          <span className="terminal-latency">{latency}ms latency</span>
        )}
      </div>
      <p className="card-subtitle">
        Test connectivity and CORS between the Vite frontend and FastAPI backend endpoints.
      </p>

      <div className="tester-actions">
        <button
          className="btn-primary"
          onClick={() => handleTestEndpoint('/health')}
          disabled={loading}
        >
          {loading && activeEndpoint === '/health' ? 'Pinging...' : 'Test GET /health'}
        </button>

        <button
          className="btn-secondary"
          onClick={() => handleTestEndpoint('/')}
          disabled={loading}
        >
          {loading && activeEndpoint === '/' ? 'Querying...' : 'Test GET /'}
        </button>
      </div>

      <div className="terminal-window">
        <div className="terminal-header">
          <div className="terminal-dots">
            <span className="terminal-dot"></span>
            <span className="terminal-dot"></span>
            <span className="terminal-dot"></span>
          </div>
          <span>HTTP {activeEndpoint} {lastCheckTime ? `(${lastCheckTime})` : ''}</span>
        </div>
        <pre className="terminal-body">
          {responseOutput || '// Click either button above to test the FastAPI backend response'}
        </pre>
      </div>
    </div>
  );
}
