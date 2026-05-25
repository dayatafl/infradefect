import React, { useCallback } from 'react';
import Header from './components/Header';
import DropZone from './components/DropZone';
import ResultPanel from './components/ResultPanel';
import { useHealth, usePredict } from './hooks/useApi';

export default function App() {
  const { status, loading: healthLoading } = useHealth();
  const { result, loading, error, predict, reset } = usePredict();

  const handleFile = useCallback((file) => {
    reset();
    predict(file);
  }, [predict, reset]);

  return (
    <div style={styles.app} className="app-container">
      <Header status={status} loading={healthLoading} />

      <main style={styles.main} className="app-main">
        {/* Top metrics bar */}
        <div style={styles.metricsBar} className="metrics-bar">
          <MetricChip label="MODEL" value="SegFormer-B2" />
          <MetricChip label="BACKBONE" value="MiT-B2" />
          <MetricChip label="INPUT" value="512 × 512" />
          <MetricChip label="CLASSES" value="5" />
          <MetricChip label="THRESHOLD" value="0.45" />
          <MetricChip
            label="DEVICE"
            value={status?.device
              ? status.device.replace('NVIDIA ', '').replace('GeForce ', '')
              : '—'}
            highlight={!!status?.device}
          />
        </div>

        {/* Main workspace */}
        <div style={styles.workspace} className="workspace">
          <div style={styles.inputCol} className="input-col">
            <DropZone onFile={handleFile} loading={loading} />

            {/* Instructions */}
            <div style={styles.instructions} className="instructions-box">
              <div style={styles.instrTitle}>HOW TO USE</div>
              <div style={styles.instrStep}>
                <span style={styles.instrNum}>01</span>
                <span>Drop or select a JPEG/PNG image of a structure</span>
              </div>
              <div style={styles.instrStep}>
                <span style={styles.instrNum}>02</span>
                <span>The model detects defects pixel-by-pixel</span>
              </div>
              <div style={styles.instrStep}>
                <span style={styles.instrNum}>03</span>
                <span>Review the overlay, severity scores, and class breakdown</span>
              </div>
            </div>
          </div>

          <div style={styles.resultCol}>
            <ResultPanel result={result} error={error} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={styles.footer} className="app-footer">
        <span>INFRADEFECT — SegFormer-B2 fine-tuned on CODEBRIM</span>
        <span style={styles.footerSep}>·</span>
        <span>
          API{' '}
          <a href="http://localhost:8000/api/docs"
            target="_blank" rel="noreferrer" style={styles.footerLink}>
            HuggingFace Space
          </a>
        </span>
      </footer>
    </div>
  );
}

function MetricChip({ label, value, highlight }) {
  return (
    <div style={styles.chip}>
      <span style={styles.chipLabel}>{label}</span>
      <span style={{ ...styles.chipValue, color: highlight ? '#f5a623' : '#7a9ab5' }}>
        {value}
      </span>
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#080b0f',
  },
  main: {
    flex: 1,
    maxWidth: 1400,
    width: '100%',
    margin: '0 auto',
    padding: '24px 32px',
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  metricsBar: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
  },
  chip: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    padding: '6px 14px',
    border: '1px solid #1e2d3d',
    borderRadius: 4,
    background: '#0d1117',
  },
  chipLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: 9,
    letterSpacing: '0.12em',
    color: '#3d5566',
  },
  chipValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 500,
  },
  workspace: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 24,
    flex: 1,
  },
  inputCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  resultCol: {
    display: 'flex',
    flexDirection: 'column',
  },
  instructions: {
    border: '1px solid #1e2d3d',
    borderRadius: 6,
    padding: '14px 16px',
    background: '#0d1117',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  instrTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: 9,
    letterSpacing: '0.14em',
    color: '#3d5566',
    marginBottom: 4,
  },
  instrStep: {
    display: 'flex',
    gap: 14,
    alignItems: 'flex-start',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#7a9ab5',
    lineHeight: 1.5,
  },
  instrNum: {
    color: '#f5a623',
    flexShrink: 0,
    fontSize: 10,
    fontWeight: 500,
  },
  footer: {
    borderTop: '1px solid #1e2d3d',
    padding: '12px 32px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#3d5566',
    letterSpacing: '0.06em',
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  footerSep: { color: '#1e2d3d' },
  footerLink: {
    color: '#f5a623',
    textDecoration: 'none',
  },
};
