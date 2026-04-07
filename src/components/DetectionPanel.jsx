import { PROCESSORS } from '../constants'

export default function DetectionPanel({ active, loading, onDetect, onPatch }) {
  const procKey = active?.processor || 'connected'
  const proc = PROCESSORS[procKey]
  const paramValues = active?.paramValues || {}

  const setParam = (key, val) => {
    onPatch({ paramValues: { ...paramValues, [key]: val } })
  }

  return (
    <div className="panel">
      <div className="panel-title" style={{ marginBottom: '8px' }}>Algoritmo de detección</div>
      <select className="proc-select" value={procKey}
        onChange={(e) => onPatch({ processor: e.target.value, paramValues: Object.fromEntries(PROCESSORS[e.target.value].params.map(p => [p.key, p.default])) })}>
        <option value="connected">Componentes Conectados</option>
        <option value="color_cluster">Clustering por Color</option>
      </select>
      <div className="proc-desc">{proc.desc}</div>

      {proc.params.map(p => (
        <ParamCtrl key={p.key} p={p} value={paramValues[p.key] ?? p.default}
          imgPixels={active ? active.imgWidth * active.imgHeight : 0}
          onChange={(v) => setParam(p.key, v)} />
      ))}

      <div className="pre-section-label">Preprocesado</div>
      <SimpleSlider label="Suavizado previo" min={0} max={9} step={1}
        value={active?.preBlur ?? 3} onChange={(v) => onPatch({ preBlur: v })} />
      <SimpleSlider label="Suavizado de regiones" min={0} max={15} step={1}
        value={active?.labelSmooth ?? 5} onChange={(v) => onPatch({ labelSmooth: v })} />

      <button className="btn-primary" style={{ marginTop: '8px' }}
        disabled={!active || loading} onClick={onDetect}>
        {loading ? 'Detectando…' : 'Detectar Regiones'}
      </button>
    </div>
  )
}

function ParamCtrl({ p, value, imgPixels, onChange }) {
  const dec = (p.step?.toString().split('.')[1] || '').length
  const round = (v) => parseFloat(Math.max(p.min, Math.min(p.max, v)).toFixed(dec))

  if (p.type === 'select') {
    return (
      <div className="ctrl">
        <label>{p.label}</label>
        <select value={value} onChange={(e) => onChange(parseInt(e.target.value))}>
          {p.options.map(o => <option key={o.v} value={o.v}>{o.l}</option>)}
        </select>
      </div>
    )
  }

  const pxHint = p.showPx && imgPixels > 0
    ? (() => { const px = Math.max(1, Math.round(imgPixels * value / 100)); return '≥ ' + (px >= 1000 ? (px/1000).toFixed(1)+'k' : px) + ' px' })()
    : null

  return (
    <>
      <div className="ctrl">
        <label>{p.label}</label>
        <button className="ctrl-sb" onClick={() => onChange(round(value - p.step))}>−</button>
        <input type="range" min={p.min} max={p.max} step={p.step} value={value}
          onChange={(e) => onChange(round(parseFloat(e.target.value)))} />
        <button className="ctrl-sb" onClick={() => onChange(round(value + p.step))}>+</button>
        <input type="number" className="ctrl-num" min={p.min} max={p.max} step={p.step} value={value}
          onChange={(e) => onChange(round(parseFloat(e.target.value) || p.min))} />
      </div>
      {pxHint && <div className="ctrl-px-hint">{pxHint}</div>}
    </>
  )
}

function SimpleSlider({ label, min, max, step, value, onChange }) {
  return (
    <div className="ctrl">
      <label>{label}</label>
      <button className="ctrl-sb" onClick={() => onChange(Math.max(min, value - step))}>−</button>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseInt(e.target.value))} />
      <button className="ctrl-sb" onClick={() => onChange(Math.min(max, value + step))}>+</button>
      <input type="number" className="ctrl-num" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Math.max(min, Math.min(max, parseInt(e.target.value) || min)))} />
    </div>
  )
}
