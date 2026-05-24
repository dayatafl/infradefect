import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

export default function DropZone({ onFile, loading }) {
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((accepted) => {
    const file = accepted[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    onFile(file);
  }, [onFile]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    multiple: false,
    disabled: loading,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  return (
    <div style={styles.wrapper}>
      <div style={styles.label}>INPUT IMAGE</div>
      <div
        {...getRootProps()}
        style={{
          ...styles.zone,
          ...(dragActive ? styles.zoneActive : {}),
          ...(loading ? styles.zoneDisabled : {}),
        }}
      >
        <input {...getInputProps()} />

        {preview ? (
          <div style={styles.previewWrap}>
            <img src={preview} alt="preview" style={styles.previewImg} />
            {loading && (
              <div style={styles.previewOverlay}>
                <Spinner />
                <div style={styles.analyzingText}>ANALYZING...</div>
              </div>
            )}
            {!loading && (
              <div style={styles.replaceHint}>DROP NEW IMAGE TO REPLACE</div>
            )}
          </div>
        ) : (
          <div style={styles.placeholder}>
            <UploadIcon active={dragActive} />
            <div style={styles.placeholderTitle}>
              {dragActive ? 'RELEASE TO ANALYZE' : 'DROP IMAGE HERE'}
            </div>
            <div style={styles.placeholderSub}>or click to browse</div>
            <div style={styles.placeholderFormats}>JPEG · PNG · Max 50MB</div>
          </div>
        )}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 40, height: 40,
      border: '2px solid #1e2d3d',
      borderTop: '2px solid #f5a623',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    }} />
  );
}

function UploadIcon({ active }) {
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none"
      style={{ marginBottom: 16, opacity: active ? 1 : 0.4,
        transition: 'opacity 0.2s', filter: active ? 'drop-shadow(0 0 8px #f5a623)' : 'none' }}>
      <rect x="1" y="1" width="46" height="46" rx="4"
        stroke={active ? '#f5a623' : '#2a3f55'} strokeWidth="1.5" strokeDasharray="4 3"/>
      <path d="M24 32V18M18 24l6-6 6 6" stroke={active ? '#f5a623' : '#2a3f55'}
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M16 36h16" stroke={active ? '#f5a623' : '#2a3f55'}
        strokeWidth="1.5" strokeLinecap="round" opacity="0.5"/>
    </svg>
  );
}

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: 10 },
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: '0.12em',
    color: '#3d5566',
  },
  zone: {
    border: '1px solid #1e2d3d',
    borderRadius: 8,
    background: '#0d1117',
    cursor: 'pointer',
    transition: 'border-color 0.2s, background 0.2s',
    minHeight: 320,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  zoneActive: {
    borderColor: '#f5a623',
    background: '#0d100a',
    animation: 'pulse-border 1s ease-in-out infinite',
  },
  zoneDisabled: {
    cursor: 'default',
    opacity: 0.7,
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: 40,
  },
  placeholderTitle: {
    fontFamily: 'var(--font-display)',
    fontWeight: 700,
    fontSize: 15,
    letterSpacing: '0.1em',
    color: '#7a9ab5',
  },
  placeholderSub: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#3d5566',
    marginTop: 6,
  },
  placeholderFormats: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#2a3f55',
    marginTop: 20,
    letterSpacing: '0.08em',
  },
  previewWrap: {
    width: '100%',
    height: '100%',
    position: 'relative',
    minHeight: 320,
  },
  previewImg: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    display: 'block',
    maxHeight: 480,
  },
  previewOverlay: {
    position: 'absolute',
    inset: 0,
    background: 'rgba(8,11,15,0.75)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
  },
  analyzingText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    letterSpacing: '0.2em',
    color: '#f5a623',
  },
  replaceHint: {
    position: 'absolute',
    bottom: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#3d5566',
    letterSpacing: '0.08em',
    whiteSpace: 'nowrap',
    background: 'rgba(8,11,15,0.8)',
    padding: '4px 10px',
    borderRadius: 3,
  },
};
