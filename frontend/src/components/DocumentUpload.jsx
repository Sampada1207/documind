import React, { useState, useRef } from 'react';
import { uploadPdfDocument } from '../services/api';

const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

export default function DocumentUpload({ onExtractionSuccess, isBackendOnline }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorInfo, setErrorInfo] = useState(null);
  const inputRef = useRef(null);

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validateAndSetFile = (file) => {
    setErrorInfo(null);
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorInfo({
        title: 'Invalid File Format',
        message: 'Only PDF files (.pdf) are supported. Please select a document in PDF format.',
        isOcrError: false,
      });
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setErrorInfo({
        title: 'File Too Large',
        message: `File size is ${formatFileSize(file.size)}. Maximum allowed size is 10 MB.`,
        isOcrError: false,
      });
      return;
    }

    if (file.size === 0) {
      setErrorInfo({
        title: 'Empty File',
        message: 'The selected PDF is 0 bytes. Please select a valid document.',
        isOcrError: false,
      });
      return;
    }

    setSelectedFile(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setErrorInfo(null);

    try {
      const result = await uploadPdfDocument(selectedFile);
      onExtractionSuccess(result);
    } catch (err) {
      const isOcr =
        err.status === 422 ||
        (typeof err.message === 'string' && err.message.toLowerCase().includes('ocr'));

      setErrorInfo({
        title: isOcr ? 'Scanned Document (OCR Required)' : `Extraction Error (${err.status || 'Network'})`,
        message: err.message,
        isOcrError: isOcr,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleResetFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setErrorInfo(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="card upload-card">
      <div className="card-title">
        <span>Upload PDF Document</span>
        <span className="badge badge-phase">PDF Ingest & Text Extraction</span>
      </div>
      <p className="card-subtitle">
        Upload a digital PDF (up to 10 MB) to extract page-by-page text in memory using PyMuPDF.
      </p>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        id="pdf-upload-input"
        accept=".pdf,application/pdf"
        style={{ display: 'none' }}
        onChange={handleChange}
        aria-label="Upload PDF file"
      />

      {/* Drag & drop dropzone */}
      <div
        className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${selectedFile ? 'dropzone-has-file' : ''}`}
        role="button"
        tabIndex={0}
        aria-label={selectedFile ? `Selected file: ${selectedFile.name}` : "Upload PDF document dropzone. Click or press Enter to browse files."}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !selectedFile && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!selectedFile && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        {!selectedFile ? (
          <div className="dropzone-content">
            <div className="dropzone-icon">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="12" y1="18" x2="12" y2="12"></line>
                <line x1="9" y1="15" x2="12" y2="12"></line>
                <line x1="15" y1="15" x2="12" y2="12"></line>
              </svg>
            </div>
            <p className="dropzone-primary-text">
              <strong>Click to upload</strong> or drag and drop your PDF here
            </p>
            <p className="dropzone-hint">
              Supports digital PDF documents up to 10 MB. In-memory processing only.
            </p>
          </div>
        ) : (
          <div className="file-preview-card">
            <div className="file-info-group">
              <div className="file-type-icon">PDF</div>
              <div className="file-details">
                <span className="file-name" title={selectedFile.name}>
                  {selectedFile.name}
                </span>
                <span className="file-meta">
                  {formatFileSize(selectedFile.size)} • Ready for extraction
                </span>
              </div>
            </div>
            <button
              type="button"
              className="btn-icon-remove"
              title="Remove file"
              aria-label="Remove selected file"
              onClick={handleResetFile}
              disabled={loading}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {/* Error Message */}
      {errorInfo && (
        <div className={`alert-banner ${errorInfo.isOcrError ? 'alert-warning' : 'alert-danger'}`} role="alert" aria-live="polite">
          <div className="alert-icon">
            {errorInfo.isOcrError ? '⚠️' : '✕'}
          </div>
          <div className="alert-content">
            <strong>{errorInfo.title}</strong>
            <p>{errorInfo.message}</p>
            {errorInfo.isOcrError && (
              <small className="alert-tip">
                Tip: Scanned or photo-based PDFs have images of text rather than digital text characters. OCR is planned for a future phase.
              </small>
            )}
          </div>
        </div>
      )}

      {/* Action Bar */}
      <div className="upload-actions">
        <button
          className="btn-primary"
          onClick={handleUpload}
          disabled={!selectedFile || loading || isBackendOnline === 'offline'}
        >
          {loading ? (
            <>
              <span className="btn-spinner"></span> Extracting Text...
            </>
          ) : (
            'Extract PDF Text'
          )}
        </button>

        {selectedFile && !loading && (
          <button className="btn-secondary" onClick={() => inputRef.current?.click()}>
            Change File
          </button>
        )}

        {isBackendOnline === 'offline' && (
          <span className="offline-warning-hint">
            ⚠️ Start backend server (port 8000) to upload documents
          </span>
        )}
      </div>
    </div>
  );
}
