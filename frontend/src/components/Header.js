import React from 'react';

const CLASS_META = {
  efflorescence: { hex: '#ffff64', label: 'Efflorescence' },
  corrosion:     { hex: '#ffa500', label: 'Corrosion' },
  crack:         { hex: '#ff4646', label: 'Crack' },
  spalling:      { hex: '#5082ff', label: 'Spalling' },
  exposed_bars:  { hex: '#32c832', label: 'Exposed Bars' },
};

export default function Header({ status, loading }) {
  const online  = !!status;
  const mode    = status?.engine_mode;
  const isModel = mode === 'model';

  return (
    <header style={styles.header}>
      {/* Scanline effect */}
      <div style={styles.scanline} />

      <div style={styles.inner}>
        {/* Logo */}
        <div style={styles.logo}>
          <div style={styles.logoMark}>
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect x="1" y="1" width="26" height="26" rx="2" stroke="#f5a623" strokeWidth="1.5"/>
              <path d="M5 14h4M19 14h4M14 5v4M14 19v4" stroke="#f5a623" strokeWidth="1.5" strokeLinecap="round"/>
              <circle cx="14" cy="14" r="3" fill="#f5a623"/>
              <path d="M8 8l2.5 2.5M17.5 17.5L20 20M20 8l-2.5 2.5M10.5 17.5L8 20" stroke="#f5a623" strokeWidth="1" strokeLinecap="round" opacity="0.5"/>
            </svg>
          </div>
          <div>
            <div style={styles.logoTitle}>INFRADEFECT</div>
            <div style={styles.logoSub}>Structural Analysis System v1.0</div>
          </div>
        </div>

        {/* Legend */}
        <div style={styles.legend}>
          {Object.entries(CLASS_META).map(([key, { hex, label }]) => (
            <div key={key} style={styles.legendItem}>
              <div style={{ ...styles.legendDot, background: hex }} />
              <span style={styles.legendLabel}>{label}</span>
            </div>
          ))}
        </div>

        {/* Status */}
        <div style={styles.statusBlock}>
          {loading ? (
            <span style={styles.statusDim}>CONNECTING...</span>
          ) : (
            <>
              <div style={{ ...styles.statusDot, background: online ? '#32c832' : '#ff4646',
                boxShadow: online ? '0 0 6px #32c832' : '0 0 6px #ff4646' }} />
              <div>
                <div style={{ ...styles.statusText, color: online ? '#32c832' : '#ff4646' }}>
                  {online ? 'API ONLINE' : 'API OFFLINE'}
                </div>
                {online && (
                  <div style={styles.statusMode}>
                    {isModel ? '● MODEL' : '◌ DEMO MODE'}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

const styles = {
  header: {
    position: 'relative',
    borderBottom: '1px solid #1e2d3d',
    background: 'linear-gradient(180deg, #0d1117 0%, #080b0f 100%)',
    overflow: 'hidden',
  },
  scanline: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, transparent, #f5a62330, transparent)',
    animation: 'scanline 4s linear infinite',
    pointerEvents: 'none',
  },
  inner: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '16px 32px',
    display: 'flex',
    alignItems: 'center',
    gap: 32,
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    flexShrink: 0,
  },
  logoMark: { flexShrink: 0 },
  logoTitle: {
    fontFamily: 'var(--font-display)',
    fontWeight: 800,
    fontSize: 18,
    letterSpacing: '0.12em',
    color: '#e8edf2',
  },
  logoSub: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#3d5566',
    letterSpacing: '0.08em',
    marginTop: 2,
  },
  legend: {
    display: 'flex',
    gap: 20,
    flex: 1,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
  },
  legendLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#7a9ab5',
    letterSpacing: '0.06em',
  },
  statusBlock: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flexShrink: 0,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
  },
  statusText: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 500,
    letterSpacing: '0.1em',
  },
  statusMode: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    color: '#f5a623',
    letterSpacing: '0.06em',
    marginTop: 2,
  },
  statusDim: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#3d5566',
    letterSpacing: '0.1em',
  },
};
