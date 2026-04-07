import { useRef } from 'react'

export default function ImageBar({ images, activeIdx, onSwitch, onRemove, onAdd }) {
  const ref = useRef(null)
  return (
    <div className="image-bar" ref={ref}
      onDragOver={(e) => { e.preventDefault(); ref.current.style.borderColor = 'var(--accent)' }}
      onDragLeave={() => { ref.current.style.borderColor = '' }}
      onDrop={(e) => { e.preventDefault(); ref.current.style.borderColor = ''
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')); if (files.length) onAdd(files) }}>
      {images.map((s, i) => (
        <div key={i} className={`image-thumb${i === activeIdx ? ' active' : ''}`} onClick={() => onSwitch(i)}>
          <img src={s.thumbUrl} alt={s.filename} />
          <div className="thumb-label">{s.filename}</div>
          {s.regions.length > 0 && <div className="thumb-badge">{s.regions.length}</div>}
          <button className="thumb-close" onClick={(e) => { e.stopPropagation(); onRemove(i) }}>×</button>
        </div>
      ))}
      <label className="add-image-btn" title="Agregar imagen">
        <span>+</span>
        <span className="add-label">Agregar</span>
        <input type="file" accept="image/*" multiple hidden
          onChange={(e) => { onAdd(Array.from(e.target.files)); e.target.value = '' }} />
      </label>
    </div>
  )
}
