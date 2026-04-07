import { memo } from 'react'
import { regionColor } from '../utils'

export default function RegionsPanel({ active, perfMs, showNumbers, setShowNumbers, mergeMode, mergeSelection,
  onToggleMergeMode, onToggleMergeItem, onDoMerge, onCancelMerge, onPreset, onSetHeight, onSetHeightWithUndo, onHover }) {
  const totalPx = active.regions.reduce((s, r) => s + r.pixelCount, 0)
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">{active.regions.length} regiones</span>
        {perfMs != null && <span className="perf-badge">{perfMs}ms</span>}
      </div>

      <div className="toggle-row">
        <input type="checkbox" id="chkNum" checked={showNumbers} onChange={(e) => setShowNumbers(e.target.checked)} />
        <label htmlFor="chkNum">Mostrar números en canvas</label>
      </div>

      {mergeMode && (
        <div className="merge-bar">
          <span className="info">{mergeSelection.length} regiones seleccionadas</span>
          <button className="btn-sm merge-btn-success" disabled={mergeSelection.length < 2} onClick={onDoMerge}>Fusionar</button>
          <button className="btn-sm" onClick={onCancelMerge}>Cancelar</button>
        </div>
      )}

      <div className="presets-row">
        {[['light-high','Claro↑'],['dark-high','Oscuro↑'],['by-area','Por área'],['flat','Plano']].map(([m, l]) => (
          <button key={m} className="btn-sm" onClick={() => onPreset(m)}>{l}</button>
        ))}
        <div style={{ flex: 1 }} />
        <button className="btn-sm merge-btn-accent" onClick={onToggleMergeMode}>
          {mergeMode ? 'Cancelar fusión' : 'Fusionar'}
        </button>
      </div>

      <div className="regions-area">
        {active.regions.map((r, idx) => {
          const pct = ((r.pixelCount / totalPx) * 100).toFixed(1)
          const c = regionColor(idx)
          const sel = mergeSelection.includes(r.id)
          return (
            <RegionRow key={r.id} r={r} idx={idx} pct={pct} c={c} sel={sel} mergeMode={mergeMode}
              onHoverIn={() => onHover(r.id)} onHoverOut={() => onHover(null)}
              onToggleMerge={() => onToggleMergeItem(r.id)}
              onSetHeight={(h) => onSetHeight(r.id, h)}
              onSetHeightWithUndo={(h) => onSetHeightWithUndo(r.id, h)} />
          )
        })}
      </div>
    </div>
  )
}

const RegionRow = memo(function RegionRow({ r, idx, pct, c, sel, mergeMode, onHoverIn, onHoverOut, onToggleMerge, onSetHeight, onSetHeightWithUndo }) {
  const grayBg = `rgb(${r.height},${r.height},${r.height})`
  return (
    <div className={`region-row${sel ? ' merge-selected' : ''}`}
      style={{ borderLeftColor: `rgb(${c[0]},${c[1]},${c[2]})` }}
      onMouseEnter={onHoverIn} onMouseLeave={onHoverOut}
      onClick={(e) => { if (mergeMode && e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') onToggleMerge() }}>
      <span className="region-num" style={{ borderColor: `rgb(${c[0]},${c[1]},${c[2]})` }}>#{idx}</span>
      <div className="swatch" style={{ background: `rgb(${r.color[0]},${r.color[1]},${r.color[2]})` }} title={`RGB(${r.color.join(',')})`} />
      <span className="area-pct">{pct}%</span>
      <input type="range" className="h-slider" min={0} max={255} value={r.height}
        style={{ accentColor: `rgb(${c[0]},${c[1]},${c[2]})` }}
        onChange={(e) => onSetHeight(parseInt(e.target.value))} />
      <div className="swatch" style={{ background: grayBg }} />
      <button className="step-btn" onClick={(e) => { e.stopPropagation(); onSetHeightWithUndo(r.height - 1) }}>−</button>
      <input type="number" className="h-num" min={0} max={255} value={r.height}
        onChange={(e) => onSetHeightWithUndo(parseInt(e.target.value) || 0)} />
      <button className="step-btn" onClick={(e) => { e.stopPropagation(); onSetHeightWithUndo(r.height + 1) }}>+</button>
    </div>
  )
})
