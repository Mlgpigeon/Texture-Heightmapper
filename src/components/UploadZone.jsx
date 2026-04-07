import { useState, useRef } from 'react'

export default function UploadZone({ onFiles }) {
  const [drag, setDrag] = useState(false)
  const fileRef = useRef(null)
  return (
    <div className="upload-area">
      <div className={`upload-zone${drag ? ' dragover' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault(); setDrag(false)
          const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
          if (files.length) onFiles(files)
        }}>
        <div className="icon">🖼️</div>
        <div className="label">Arrastra texturas aquí o haz clic para seleccionar</div>
        <div className="hint">PNG, JPG — cualquier tamaño · múltiples imágenes · procesamiento 100% local</div>
      </div>
      <input ref={fileRef} type="file" accept="image/*" multiple hidden
        onChange={(e) => { onFiles(Array.from(e.target.files)); e.target.value = '' }} />
    </div>
  )
}
