import React, { useState } from 'react';

const CLASS_META = {
  efflorescence: { hex: '#ffff64', label: 'Efflorescence' },
  corrosion:     { hex: '#ffa500', label: 'Corrosion' },
  crack:         { hex: '#ff4646', label: 'Crack' },
  spalling:      { hex: '#5082ff', label: 'Spalling' },
  exposed_bars:  { hex: '#32c832', label: 'Exposed Bars' },
};

const SEVERITY_COLOR = {
  none:     '#3d5566',
  low:      '#f5a623',
  medium:   '#ff8c00',
  high:     '#ff4646',
  critical: '#ff0000',
};

export default function ResultPanel({ result, error }) {
  const [view, setView] = useState('overlay'); // 'overlay' | 'mask'

  if (error) {
    return (
      <div style={styles.wrapper} className="result-wrapper">
        <div style={styles.label}>ANALYSIS RESULT</div>
        <div style={styles.errorBox} className="error-box">
          <div style={styles.errorIcon}>⚠</div>
          <div style={styles.errorText}>{error}</div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div style={styles.wrapper} className="result-wrapper">
        <div style={styles.label}>ANALYSIS RESULT</div>
        <div style={styles.emptyBox} className="empty-box">
          <div style={styles.emptyGrid}>
            {[...Array(12)].map((_, i) => (
              <div key={i} style={{ ...styles.emptyCell,
                opacity: 0.03 + (i % 4) * 0.02 }} />
            ))}
          </div>
          <div style={styles.emptyText}>AWAITING INPUT</div>
          <div style={styles.emptySubtext}>Submit an image to begin structural analysis</div>
        </div>
      </div>
    );
  }

  const { overlay_b64, mask_b64, class_stats, inference_ms, engine_mode, image_size, job_id } = result;
  const defects = Object.entries(class_stats).filter(([, s]) => s.severity !== 'none');
  const imgSrc  = `data:image/png;base64,${overlay_b64}`;
  const maskSrc = mask_b64 ? `data:image/png;base64,${mask_b64}` : imgSrc;

  return (
    <div style={{ ...styles.wrapper, animation: 'fadeUp 0.4s ease' }}>
      <div style={styles.label}>ANALYSIS RESULT</div>

      {/* Image viewer */}
      <div style={styles.imageCard} className="image-card">
        <div style={styles.imageToolbar} className="image-toolbar">
          <div style={styles.jobId}>
            <span style={styles.dimText}>JOB</span>
            <span style={styles.monoText}>{job_id}</span>
          </div>
          <div style={styles.viewToggle}>
            <button style={{ ...styles.toggleBtn, ...(view === 'overlay' ? styles.toggleActive : {}) }}
              onClick={() => setView('overlay')}>OVERLAY</button>
            <button style={{ ...styles.toggleBtn, ...(view === 'mask' ? styles.toggleActive : {}) }}
              onClick={() => setView('mask')}>MASK</button>
          </div>
          <div style={styles.imageMeta}>
            <span style={styles.monoText}>{image_size[0]}×{image_size[1]}</span>
          </div>
        </div>
        <div style={styles.imageWrap}>
          <img
            src={view === 'overlay' ? imgSrc : maskSrc}
            alt="result"
            style={styles.resultImg}
          />
        </div>
      </div>

      {/* Stats grid */}
      <div style={styles.statsGrid} className="stats-grid">
        {Object.entries(class_stats).map(([name, stat]) => {
          const meta = CLASS_META[name] || { hex: '#888', label: name };
          const pct = stat.percentage;
          const maxPct = 30;
          const barW = Math.min((pct / maxPct) * 100, 100);
          const isActive = stat.severity !== 'none';

          return (
            <div key={name} style={{ ...styles.statCard, ...(isActive ? styles.statCardActive : {}) }}>
              <div style={styles.statHeader}>
                <div style={{ ...styles.statDot, background: meta.hex,
                  boxShadow: isActive ? `0 0 6px ${meta.hex}` : 'none' }} />
                <span style={styles.statName}>{meta.label.toUpperCase()}</span>
                <span style={{ ...styles.statSeverity,
                  color: SEVERITY_COLOR[stat.severity] || '#888' }}>
                  {stat.severity.toUpperCase()}
                </span>
              </div>
              <div style={styles.statBar}>
                <div style={{ ...styles.statBarFill,
                  width: `${barW}%`,
                  background: isActive ? meta.hex : '#1e2d3d',
                  animation: 'progress-fill 0.8s ease',
                }} />
              </div>
              <div style={styles.statFooter}>
                <span style={styles.statPct}>{pct.toFixed(2)}%</span>
                <span style={styles.statPx}>{stat.pixel_count.toLocaleString()} px</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary footer */}
      <div style={styles.footer} className="result-footer">
        <div style={styles.footerItem}>
          <span style={styles.dimText}>INFERENCE</span>
          <span style={styles.monoText}>{inference_ms}ms</span>
        </div>
        <div style={styles.footerItem}>
          <span style={styles.dimText}>ENGINE</span>
          <span style={{ ...styles.monoText,
            color: engine_mode === 'model' ? '#32c832' : '#f5a623' }}>
            {engine_mode.toUpperCase()}
          </span>
        </div>
        <div style={styles.footerItem}>
          <span style={styles.dimText}>DEFECTS</span>
          <span style={{ ...styles.monoText,
            color: defects.length > 0 ? '#ff4646' : '#32c832' }}>
            {defects.length > 0
              ? defects.map(([n]) => CLASS_META[n]?.label || n).join(', ')
              : 'NONE DETECTED'}
          </span>
        </div>
      </div>
    </div>
  );
}

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: 12 },
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: '0.12em',
    color: '#3d5566',
  },
  errorBox: {
    border: '1px solid #ff464640',
    borderRadius: 8,
    background: '#0d0a0a',
    padding: 32,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
  },
  errorIcon: { fontSize: 28, color: '#ff4646' },
  errorText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    color: '#ff8080',
    textAlign: 'center',
    lineHeight: 1.6,
  },
  emptyBox: {
    border: '1px solid #1e2d3d',
    borderRadius: 8,
    background: '#0d1117',
    minHeight: 320,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
    gap: 10,
  },
  emptyGrid: {
    position: 'absolute',
    inset: 0,
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gridTemplateRows: 'repeat(3, 1fr)',
    gap: 1,
    pointerEvents: 'none',
  },
  emptyCell: {
    background: '#7a9ab5',
  },
  emptyText: {
    fontFamily: 'var(--font-display)',
    fontWeight: 700,
    fontSize: 14,
    letterSpacing: '0.15em',
    color: '#2a3f55',
    position: 'relative',
  },
  emptySubtext: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#1e2d3d',
    position: 'relative',
  },
  imageCard: {
    border: '1px solid #1e2d3d',
    borderRadius: 8,
    overflow: 'hidden',
    background: '#080b0f',
  },
  imageToolbar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 14px',
    borderBottom: '1px solid #1e2d3d',
    background: '#0d1117',
  },
  jobId: { display: 'flex', gap: 8, alignItems: 'center' },
  viewToggle: {
    display: 'flex',
    border: '1px solid #1e2d3d',
    borderRadius: 4,
    overflow: 'hidden',
  },
  toggleBtn: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: '0.08em',
    padding: '5px 12px',
    border: 'none',
    background: 'transparent',
    color: '#3d5566',
    cursor: 'pointer',
    transition: 'background 0.15s, color 0.15s',
  },
  toggleActive: {
    background: '#1a2330',
    color: '#f5a623',
  },
  imageMeta: {},
  imageWrap: {
    background: '#060809',
    display: 'flex',
    justifyContent: 'center',
  },
  resultImg: {
    maxWidth: '100%',
    maxHeight: 440,
    objectFit: 'contain',
    display: 'block',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 8,
  },
  statCard: {
    border: '1px solid #1e2d3d',
    borderRadius: 6,
    padding: '12px 14px',
    background: '#0d1117',
    transition: 'border-color 0.2s',
  },
  statCardActive: {
    borderColor: '#1e2d3d',
    background: '#0d1117',
  },
  statHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  statDot: {
    width: 8, height: 8,
    borderRadius: '50%',
    flexShrink: 0,
    transition: 'box-shadow 0.3s',
  },
  statName: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    letterSpacing: '0.08em',
    color: '#7a9ab5',
    flex: 1,
  },
  statSeverity: {
    fontFamily: 'var(--font-mono)',
    fontSize: 9,
    letterSpacing: '0.1em',
    fontWeight: 500,
  },
  statBar: {
    height: 3,
    background: '#1a2330',
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 8,
  },
  statBarFill: {
    height: '100%',
    borderRadius: 2,
    transition: 'width 0.8s ease',
  },
  statFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statPct: {
    fontFamily: 'var(--font-mono)',
    fontSize: 14,
    fontWeight: 500,
    color: '#e8edf2',
  },
  statPx: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#3d5566',
  },
  footer: {
    display: 'flex',
    gap: 24,
    padding: '12px 16px',
    border: '1px solid #1e2d3d',
    borderRadius: 6,
    background: '#0d1117',
    flexWrap: 'wrap',
  },
  footerItem: {
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  },
  dimText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#3d5566',
    letterSpacing: '0.1em',
  },
  monoText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#7a9ab5',
    letterSpacing: '0.05em',
  },
};