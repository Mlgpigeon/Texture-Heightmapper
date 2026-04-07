import { useReducer, useRef, useEffect, useCallback, useState } from 'react'
import { typedArrayToBase64, base64ToInt32Array, imgRGBAtoDataURL } from './utils'
import { reducer, INIT, makeImageState } from './reducer'
import { renderToCanvas } from './canvasRenderer'
import { hitTestLabel, drawLabelsOnOverlay } from './labelsOverlay'
import UploadZone from './components/UploadZone'
import ImageBar from './components/ImageBar'
import DetectionPanel from './components/DetectionPanel'
import RegionsPanel from './components/RegionsPanel'

export default function App() {
  const [st, dispatch] = useReducer(reducer, INIT)
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('Procesando...')
  const [toastMsg, setToastMsg] = useState(null)
  const [showNumbers, setShowNumbers] = useState(true)
  const [zoomLabel, setZoomLabel] = useState('100%')
  const [mergeMode, setMergeMode] = useState(false)
  const [mergeSelection, setMergeSelection] = useState([])
  const [perfMs, setPerfMs] = useState(null)
  const [rightWidth, setRightWidth] = useState(400)

  const canvasRef = useRef(null)
  const labelsCanvasRef = useRef(null)
  const viewportRef = useRef(null)
  const posRef = useRef(null)
  const workerRef = useRef(null)
  const toastTimer = useRef(null)
  const renderTimer = useRef(null)
  const hoverRef = useRef(null)
  const showNumRef = useRef(showNumbers)
  const zoomRef = useRef({ level: 1, panX: 0, panY: 0 })
  const dragRef = useRef({ active: false, sx: 0, sy: 0, px: 0, py: 0 })
  const touchRef = useRef({ dist: 0 })
  const resizeDragRef = useRef({ active: false, startX: 0, startWidth: 0 })
  const mergeSelRef = useRef([])
  const mergeModeRef = useRef(false)
  const activeRef = useRef(null)
  const boxSelRef = useRef(null)
  const boxElRef = useRef(null)

  const { images, activeIdx } = st
  const active = activeIdx >= 0 ? images[activeIdx] : null

  // Keep refs in sync
  useEffect(() => { showNumRef.current = showNumbers }, [showNumbers])
  useEffect(() => { mergeModeRef.current = mergeMode }, [mergeMode])
  useEffect(() => { activeRef.current = active })
  useEffect(() => { mergeSelRef.current = mergeSelection }, [mergeSelection])

  // ── Toast helper ──
  const toast = useCallback((msg) => {
    clearTimeout(toastTimer.current)
    setToastMsg(null)
    requestAnimationFrame(() => {
      setToastMsg(msg)
      toastTimer.current = setTimeout(() => setToastMsg(null), 3100)
    })
  }, [])

  // ── Render helpers ──
  const renderView = useCallback((highlightId) => {
    const s = activeIdx >= 0 ? images[activeIdx] : null
    const hi = highlightId ?? hoverRef.current
    renderToCanvas(canvasRef.current, s, hi)
    const { level, panX, panY } = zoomRef.current
    drawLabelsOnOverlay(labelsCanvasRef.current, viewportRef.current, s, showNumbers, hi, level, panX, panY, mergeSelRef.current)
  }, [images, activeIdx, showNumbers])

  // Re-render when state changes — also when mergeSelection changes (label highlight)
  useEffect(() => { renderView() }, [renderView, mergeSelection])

  // Re-draw labels when the right panel is resized (viewport changes width)
  useEffect(() => {
    const s = activeIdx >= 0 ? images[activeIdx] : null
    const { level, panX, panY } = zoomRef.current
    drawLabelsOnOverlay(labelsCanvasRef.current, viewportRef.current, s, showNumbers, hoverRef.current, level, panX, panY, mergeSelRef.current)
  }, [rightWidth, images, activeIdx, showNumbers])

  // ── Apply zoom transform ──
  const applyZoom = useCallback(() => {
    const { level, panX, panY } = zoomRef.current
    if (posRef.current) posRef.current.style.transform = `translate(${panX}px,${panY}px) scale(${level})`
    setZoomLabel(Math.round(level * 100) + '%')
    const s = activeIdx >= 0 ? images[activeIdx] : null
    drawLabelsOnOverlay(labelsCanvasRef.current, viewportRef.current, s, showNumRef.current, hoverRef.current, level, panX, panY, mergeSelRef.current)
    clearTimeout(renderTimer.current)
    renderTimer.current = setTimeout(() => renderView(), 80)
  }, [renderView, images, activeIdx])

  const clampPan = useCallback(() => {
    const s = active; if (!s || !viewportRef.current) return
    const vw = viewportRef.current.clientWidth, vh = viewportRef.current.clientHeight
    const { level } = zoomRef.current
    const sw = s.imgWidth * level, sh = s.imgHeight * level
    zoomRef.current.panX = Math.max(Math.min(0, vw - sw), Math.min(0, zoomRef.current.panX))
    zoomRef.current.panY = Math.max(Math.min(0, vh - sh), Math.min(0, zoomRef.current.panY))
  }, [active])

  const adjustZoom = useCallback((delta, cx, cy) => {
    const vp = viewportRef.current; if (!vp) return
    const vw = vp.clientWidth, vh = vp.clientHeight
    const ox = cx ?? vw/2, oy = cy ?? vh/2
    const prev = zoomRef.current.level
    const next = Math.max(.1, Math.min(10, prev + delta))
    const ratio = next / prev
    zoomRef.current.panX = ox - ratio * (ox - zoomRef.current.panX)
    zoomRef.current.panY = oy - ratio * (oy - zoomRef.current.panY)
    zoomRef.current.level = next
    clampPan(); applyZoom()
  }, [clampPan, applyZoom])

  const zoomFit = useCallback(() => {
    const s = active; const vp = viewportRef.current; if (!s || !vp) return
    const vw = vp.clientWidth, vh = vp.clientHeight
    const level = Math.min(vw / s.imgWidth, vh / s.imgHeight, 1)
    zoomRef.current = {
      level, panX: Math.max(0, (vw - s.imgWidth * level) / 2),
      panY: Math.max(0, (vh - s.imgHeight * level) / 2)
    }
    applyZoom()
  }, [active, applyZoom])

  // ── Worker ──
  useEffect(() => {
    const w = new Worker('./worker.js')
    workerRef.current = w
    w.onmessage = (e) => {
      const msg = e.data
      if (msg.type === 'result') {
        dispatch({ type: 'PUSH_UNDO' })
        dispatch({ type: 'PATCH_ACTIVE', patch: {
          labelMap: new Int32Array(msg.labelMap),
          regions: msg.regions,
          viewMode: 'regions',
        }})
        setLoading(false)
        setPerfMs(msg.elapsed)
        toast(`${msg.regions.length} regiones detectadas en ${msg.elapsed}ms`)
      } else if (msg.type === 'error') {
        setLoading(false)
        toast('Error: ' + msg.message)
      }
    }
    return () => w.terminate()
  }, [toast])

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') {
        e.preventDefault()
        dispatch({ type: 'REDO' })
        toast('Rehacer →')
        return
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        dispatch({ type: 'UNDO' })
        toast('Deshacer ←')
        return
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault()
        dispatch({ type: 'REDO' })
        toast('Rehacer →')
        return
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === '=' || e.key === '+')) { e.preventDefault(); adjustZoom(.2) }
      if ((e.ctrlKey || e.metaKey) && e.key === '-') { e.preventDefault(); adjustZoom(-.2) }
      if ((e.ctrlKey || e.metaKey) && e.key === '0') { e.preventDefault(); zoomFit() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [adjustZoom, zoomFit, toast])

  // ── Zoom fit on image switch ──
  useEffect(() => {
    zoomRef.current = { level: 1, panX: 0, panY: 0 }
    if (active && viewportRef.current) {
      requestAnimationFrame(() => zoomFit())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIdx])

  // ── Wheel zoom ──
  useEffect(() => {
    const vp = viewportRef.current; if (!vp) return
    const onWheel = (e) => {
      e.preventDefault()
      const rect = vp.getBoundingClientRect()
      const delta = e.deltaY < 0 ? .15 : -.15
      adjustZoom(delta, e.clientX - rect.left, e.clientY - rect.top)
    }
    vp.addEventListener('wheel', onWheel, { passive: false })
    return () => vp.removeEventListener('wheel', onWheel)
  }, [adjustZoom])

  // ── Mouse drag pan ──
  useEffect(() => {
    const vp = viewportRef.current; if (!vp) return
    const down = (e) => {
      if (e.button !== 0) return
      // Shift+drag in merge mode → box select, don't pan
      if (e.shiftKey && mergeModeRef.current) return
      dragRef.current = { active: true, sx: e.clientX, sy: e.clientY, px: zoomRef.current.panX, py: zoomRef.current.panY }
      vp.classList.add('dragging')
    }
    const move = (e) => {
      if (!dragRef.current.active) return
      zoomRef.current.panX = dragRef.current.px + (e.clientX - dragRef.current.sx)
      zoomRef.current.panY = dragRef.current.py + (e.clientY - dragRef.current.sy)
      clampPan(); applyZoom()
    }
    const up = () => { if (!dragRef.current.active) return; dragRef.current.active = false; vp.classList.remove('dragging') }
    vp.addEventListener('mousedown', down)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { vp.removeEventListener('mousedown', down); window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [clampPan, applyZoom])

  // ── Label hover highlight & click-to-merge & box select on canvas ──
  useEffect(() => {
    const vp = viewportRef.current; if (!vp) return
    let prevLabelIdx = -1

    // — Hover highlight —
    const onMove = (e) => {
      // Box select drag in progress
      if (boxSelRef.current) {
        const rect = vp.getBoundingClientRect()
        boxSelRef.current.ex = e.clientX - rect.left
        boxSelRef.current.ey = e.clientY - rect.top
        const b = boxSelRef.current, el = boxElRef.current
        if (el) {
          const x = Math.min(b.sx, b.ex), y = Math.min(b.sy, b.ey)
          const w = Math.abs(b.ex - b.sx), h = Math.abs(b.ey - b.sy)
          el.style.display = 'block'
          el.style.left = x + 'px'; el.style.top = y + 'px'
          el.style.width = w + 'px'; el.style.height = h + 'px'
        }
        return
      }

      if (dragRef.current.active) return
      const s = activeRef.current
      if (!s?.regions?.length) return
      const rect = vp.getBoundingClientRect()
      const mx = e.clientX - rect.left, my = e.clientY - rect.top
      const { level, panX, panY } = zoomRef.current
      const idx = hitTestLabel(vp, s, showNumRef.current, level, panX, panY, mx, my)

      if (idx === prevLabelIdx) return
      prevLabelIdx = idx

      if (idx >= 0) {
        vp.style.cursor = 'pointer'
        const regionId = s.regions[idx].id
        hoverRef.current = regionId
        renderToCanvas(canvasRef.current, s, regionId)
        drawLabelsOnOverlay(labelsCanvasRef.current, vp, s, showNumRef.current, regionId, level, panX, panY, mergeSelRef.current)
      } else {
        vp.style.cursor = ''
        if (hoverRef.current != null) {
          hoverRef.current = null
          renderToCanvas(canvasRef.current, s, null)
          drawLabelsOnOverlay(labelsCanvasRef.current, vp, s, showNumRef.current, null, level, panX, panY, mergeSelRef.current)
        }
      }
    }

    // — Click on label to toggle merge —
    const onClick = (e) => {
      if (!mergeModeRef.current) return
      const d = dragRef.current
      if (Math.abs(e.clientX - d.sx) > 4 || Math.abs(e.clientY - d.sy) > 4) return
      const s = activeRef.current
      if (!s?.regions?.length) return
      const rect = vp.getBoundingClientRect()
      const mx = e.clientX - rect.left, my = e.clientY - rect.top
      const { level, panX, panY } = zoomRef.current
      const idx = hitTestLabel(vp, s, showNumRef.current, level, panX, panY, mx, my)
      if (idx >= 0) {
        setMergeSelection(prev => {
          const id = s.regions[idx].id
          return prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        })
      }
    }

    // — Box select: Shift+drag in merge mode —
    const onDown = (e) => {
      if (e.button !== 0 || !e.shiftKey || !mergeModeRef.current) return
      const rect = vp.getBoundingClientRect()
      const mx = e.clientX - rect.left, my = e.clientY - rect.top
      boxSelRef.current = { sx: mx, sy: my, ex: mx, ey: my }
      e.preventDefault()
    }

    const onUp = (e) => {
      if (!boxSelRef.current) return
      const b = boxSelRef.current
      boxSelRef.current = null
      // Hide rectangle
      if (boxElRef.current) boxElRef.current.style.display = 'none'

      // Find all label centers inside the box
      const s = activeRef.current
      if (!s?.regions?.length) return
      const { level, panX, panY } = zoomRef.current
      const x1 = Math.min(b.sx, b.ex), y1 = Math.min(b.sy, b.ey)
      const x2 = Math.max(b.sx, b.ex), y2 = Math.max(b.sy, b.ey)
      // Ignore tiny drags (< 8px)
      if (x2 - x1 < 8 && y2 - y1 < 8) return

      const idsInBox = []
      for (let i = 0; i < s.regions.length; i++) {
        const r = s.regions[i]
        const cx = (r.bbox[0] + r.bbox[2]) / 2 * level + panX
        const cy = (r.bbox[1] + r.bbox[3]) / 2 * level + panY
        if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) {
          idsInBox.push(r.id)
        }
      }
      if (idsInBox.length) {
        setMergeSelection(prev => {
          const s = new Set(prev)
          idsInBox.forEach(id => s.add(id))
          return [...s]
        })
      }
    }

    vp.addEventListener('mousedown', onDown)
    vp.addEventListener('mousemove', onMove)
    vp.addEventListener('click', onClick)
    window.addEventListener('mouseup', onUp)
    return () => {
      vp.removeEventListener('mousedown', onDown)
      vp.removeEventListener('mousemove', onMove)
      vp.removeEventListener('click', onClick)
      window.removeEventListener('mouseup', onUp)
    }
  }, [clampPan, applyZoom])

  // ── Touch ──
  useEffect(() => {
    const vp = viewportRef.current; if (!vp) return
    const tstart = (e) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX
        const dy = e.touches[0].clientY - e.touches[1].clientY
        touchRef.current.dist = Math.hypot(dx, dy)
      } else {
        dragRef.current = { active: true, sx: e.touches[0].clientX, sy: e.touches[0].clientY, px: zoomRef.current.panX, py: zoomRef.current.panY }
      }
    }
    const tmove = (e) => {
      e.preventDefault()
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX
        const dy = e.touches[0].clientY - e.touches[1].clientY
        const dist = Math.hypot(dx, dy)
        const delta = (dist - touchRef.current.dist) / touchRef.current.dist
        touchRef.current.dist = dist
        adjustZoom(delta * zoomRef.current.level * .5)
      } else if (dragRef.current.active) {
        zoomRef.current.panX = dragRef.current.px + (e.touches[0].clientX - dragRef.current.sx)
        zoomRef.current.panY = dragRef.current.py + (e.touches[0].clientY - dragRef.current.sy)
        clampPan(); applyZoom()
      }
    }
    vp.addEventListener('touchstart', tstart, { passive: true })
    vp.addEventListener('touchmove', tmove, { passive: false })
    return () => { vp.removeEventListener('touchstart', tstart); vp.removeEventListener('touchmove', tmove) }
  }, [adjustZoom, clampPan, applyZoom])

  // ── Right-panel resize drag ──
  useEffect(() => {
    const onMove = (e) => {
      if (!resizeDragRef.current.active) return
      const dx = resizeDragRef.current.startX - e.clientX
      const newW = Math.max(220, Math.min(800, resizeDragRef.current.startWidth + dx))
      setRightWidth(newW)
    }
    const onUp = () => { resizeDragRef.current.active = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  // ── Load image file ──
  const loadImageFile = useCallback((file) => {
    const img = new Image()
    img.onload = () => {
      const tc = document.createElement('canvas'); tc.width = tc.height = 144
      const tx = tc.getContext('2d')
      const sc = Math.max(144 / img.width, 144 / img.height)
      tx.drawImage(img, (144 - img.width*sc)/2, (144 - img.height*sc)/2, img.width*sc, img.height*sc)
      const thumbUrl = tc.toDataURL('image/jpeg', .7)
      const tmp = document.createElement('canvas'); tmp.width = img.width; tmp.height = img.height
      const tmpCtx = tmp.getContext('2d'); tmpCtx.drawImage(img, 0, 0)
      const { data } = tmpCtx.getImageData(0, 0, img.width, img.height)
      const imgState = makeImageState(file.name, data, img.width, img.height, thumbUrl)
      dispatch({ type: 'ADD_IMAGE', img: imgState })
      URL.revokeObjectURL(img.src)
    }
    img.src = URL.createObjectURL(file)
  }, [])

  // ── Detect regions ──
  const detect = useCallback(() => {
    if (!active) return
    setLoading(true); setLoadingMsg('Detectando regiones...')
    const params = { ...active.paramValues, _pre_blur: active.preBlur, _label_smooth: active.labelSmooth }
    const copy = new Uint8Array(active.imgRGBA.length); copy.set(active.imgRGBA)
    workerRef.current.postMessage({ type: 'detect', processor: active.processor,
      width: active.imgWidth, height: active.imgHeight, rgba: copy.buffer, params }, [copy.buffer])
  }, [active])

  // ── Download heightmap ──
  const downloadHeightmap = useCallback(() => {
    if (!active?.labelMap) return
    const { imgWidth: w, imgHeight: h, labelMap: lm, regions } = active
    const maxId = regions.reduce((m, r) => Math.max(m, r.id), 0)
    const lutH = new Uint8Array(maxId + 1).fill(128)
    regions.forEach(r => { lutH[r.id] = r.height })
    const c = document.createElement('canvas'); c.width = w; c.height = h
    const cx = c.getContext('2d'); const img = cx.createImageData(w, h); const d = img.data
    for (let i = 0; i < w*h; i++) { const p=i<<2; const v = lm[i]>=0 ? lutH[lm[i]] : 0; d[p]=v; d[p+1]=v; d[p+2]=v; d[p+3]=255 }
    cx.putImageData(img, 0, 0)
    c.toBlob(blob => { const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download=active.filename.replace(/\.[^.]+$/,'')+'_heightmap.png'; a.click(); URL.revokeObjectURL(url) }, 'image/png')
  }, [active])

  // ── Save / Load project ──
  const saveProject = useCallback(() => {
    if (!images.length) { toast('No hay imágenes cargadas'); return }
    const project = { version: 1, activeIdx,
      images: images.map(s => ({
        filename: s.filename, imgData: imgRGBAtoDataURL(s.imgRGBA, s.imgWidth, s.imgHeight),
        imgWidth: s.imgWidth, imgHeight: s.imgHeight, thumbUrl: s.thumbUrl,
        labelMap: s.labelMap ? typedArrayToBase64(s.labelMap) : null,
        regions: JSON.parse(JSON.stringify(s.regions)), viewMode: s.viewMode,
        processor: s.processor, paramValues: { ...s.paramValues },
        preBlur: s.preBlur, labelSmooth: s.labelSmooth
      }))
    }
    const blob = new Blob([JSON.stringify(project)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href=url; a.download='heightmap_project.json'; a.click(); URL.revokeObjectURL(url)
    toast('Proyecto guardado ✓')
  }, [images, activeIdx, toast])

  const loadProject = useCallback((project) => {
    if (!project?.images?.length) { toast('Proyecto inválido'); return }
    let loaded = 0; const total = project.images.length; const result = new Array(total)
    project.images.forEach((d, idx) => {
      const img = new Image()
      img.onload = () => {
        const c = document.createElement('canvas'); c.width=d.imgWidth; c.height=d.imgHeight
        const cx = c.getContext('2d'); cx.drawImage(img, 0, 0)
        const { data } = cx.getImageData(0, 0, d.imgWidth, d.imgHeight)
        const s = makeImageState(d.filename, data, d.imgWidth, d.imgHeight, d.thumbUrl)
        s.labelMap = d.labelMap ? base64ToInt32Array(d.labelMap) : null
        s.regions = d.regions || []; s.viewMode = d.viewMode || 'original'
        s.processor = d.processor || 'connected'; s.paramValues = d.paramValues || {}
        s.preBlur = d.preBlur ?? 3; s.labelSmooth = d.labelSmooth ?? 5
        result[idx] = s; loaded++
        if (loaded === total) {
          const ai2 = project.activeIdx >= 0 && project.activeIdx < total ? project.activeIdx : 0
          dispatch({ type: 'LOAD', images: result, activeIdx: ai2 })
          toast(`Proyecto cargado ✓ (${total} imagen${total!==1?'es':''})`)
        }
      }
      img.src = d.imgData
    })
  }, [toast])

  // ── Merge ──
  const doMerge = useCallback(() => {
    if (mergeSelection.length < 2) return
    dispatch({ type: 'PUSH_UNDO' })
    dispatch({ type: 'MERGE', ids: mergeSelection })
    toast(`${mergeSelection.length - 1} región(es) fusionadas`)
    setMergeMode(false); setMergeSelection([])
  }, [mergeSelection, toast])

  const toggleMergeItem = useCallback((id) => {
    setMergeSelection(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }, [])

  // ── Preset ──
  const applyPreset = useCallback((mode) => {
    dispatch({ type: 'PUSH_UNDO' })
    dispatch({ type: 'APPLY_PRESET', mode })
    if (active?.viewMode === 'heightmap') setTimeout(() => renderView(), 50)
  }, [active, renderView])

  // ── Height change ──
  const setHeight = useCallback((id, h) => {
    dispatch({ type: 'SET_HEIGHT', id, h: Math.max(0, Math.min(255, h)) })
  }, [])

  const setHeightWithUndo = useCallback((id, h) => {
    dispatch({ type: 'PUSH_UNDO' })
    dispatch({ type: 'SET_HEIGHT', id, h: Math.max(0, Math.min(255, h)) })
    if (active?.viewMode === 'heightmap') setTimeout(() => renderView(), 20)
  }, [active, renderView])

  const hasImages = images.length > 0

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-brand">
          <div className="app-title">Heightmap Studio</div>
          <div className="app-subtitle">Detección de regiones → Alturas → Heightmap para displacement</div>
        </div>
        <div className="toolbar">
          <button className="btn-toolbar" onClick={saveProject}>
            <svg viewBox="0 0 16 16" fill="currentColor"><path d="M2 2a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H2zm0 1h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zm2 3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1H4zm0 1h8v4H4V7zm1.5 1a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5z"/></svg>
            Guardar
          </button>
          <label className="btn-toolbar" style={{ cursor: 'pointer' }}>
            <svg viewBox="0 0 16 16" fill="currentColor"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 1.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 2.707V11.5a.5.5 0 0 1-1 0V2.707L5.354 4.854a.5.5 0 1 1-.708-.708l3-3z"/></svg>
            Cargar
            <input type="file" accept=".json" hidden onChange={(e) => {
              const f = e.target.files[0]; if (!f) return
              e.target.value = ''
              const reader = new FileReader()
              reader.onload = (ev) => { try { loadProject(JSON.parse(ev.target.result)) } catch (err) { toast('Error: ' + err.message) } }
              reader.readAsText(f)
            }} />
          </label>
          <div className="toolbar-sep" />
          <button className="btn-toolbar" disabled={!active || active.undoStack.length === 0}
            onClick={() => { dispatch({ type: 'UNDO' }); toast('Deshacer ←') }}>
            <svg viewBox="0 0 16 16" fill="currentColor"><path fillRule="evenodd" d="M8 3a5 5 0 1 1-4.546 2.914.5.5 0 0 0-.908-.417A6 6 0 1 0 8 2v1z"/><path d="M8 4.466V.534a.25.25 0 0 0-.41-.192L5.23 2.308a.25.25 0 0 0 0 .384l2.36 1.966A.25.25 0 0 0 8 4.466z"/></svg>
            Deshacer
          </button>
          <button className="btn-toolbar" disabled={!active || active.redoStack.length === 0}
            onClick={() => { dispatch({ type: 'REDO' }); toast('Rehacer →') }}>
            <svg viewBox="0 0 16 16" fill="currentColor"><path fillRule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/><path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966a.25.25 0 0 1 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/></svg>
            Rehacer
          </button>
        </div>
      </header>

      {/* ── Upload or main layout ── */}
      {!hasImages ? (
        <UploadZone onFiles={(files) => files.forEach(loadImageFile)} />
      ) : (
        <div className="main-layout">
          {/* ── Left column ── */}
          <div className="col-left">
            <ImageBar images={images} activeIdx={activeIdx}
              onSwitch={(i) => dispatch({ type: 'SET_ACTIVE', idx: i })}
              onRemove={(i) => dispatch({ type: 'REMOVE_IMAGE', idx: i })}
              onAdd={(files) => files.forEach(loadImageFile)} />

            <div className="canvas-area">
              <div className="tab-bar">
                {['original', 'regions', 'heightmap'].map(mode => {
                  if (mode !== 'original' && (!active?.regions?.length)) return null
                  return (
                    <button key={mode} className={`tab${active?.viewMode === mode ? ' active' : ''}`}
                      onClick={() => {
                        dispatch({ type: 'PATCH_ACTIVE', patch: { viewMode: mode } })
                      }}>
                      {mode === 'original' ? 'Original' : mode === 'regions' ? 'Regiones' : 'Heightmap'}
                    </button>
                  )
                })}
                <div className="tab-spacer" />
                <button className="remove-img-btn" onClick={() => dispatch({ type: 'REMOVE_IMAGE', idx: activeIdx })}>
                  ✕ Quitar imagen
                </button>
              </div>

              <div className="zoom-bar">
                <button className="zoom-btn" onClick={() => adjustZoom(-.25)}>−</button>
                <span className="zoom-label">{zoomLabel}</span>
                <button className="zoom-btn" onClick={() => adjustZoom(.25)}>+</button>
                <button className="zoom-fit-btn" onClick={zoomFit}>Ajustar</button>
                <span style={{ fontSize: '11px', color: 'var(--text3)', marginLeft: '8px' }}>
                  Ctrl+Scroll · arrastrar
                </span>
              </div>

              <div className="canvas-viewport" ref={viewportRef}>
                <div className="canvas-positioner" ref={posRef}>
                  <canvas ref={canvasRef} />
                  {loading && (
                    <div className="loading-overlay">
                      <div className="loading-spinner" />
                      <span>{loadingMsg}</span>
                    </div>
                  )}
                </div>
                <canvas ref={labelsCanvasRef} className="labels-overlay" />
                <div ref={boxElRef} className="box-select" />
              </div>

              <div className="canvas-info">
                <span>{active ? `${active.filename} — ${active.imgWidth}×${active.imgHeight}` : '—'}</span>
                <span>{active?.regions?.length > 0 ? `${active.regions.length} regiones` : ''}</span>
              </div>
            </div>

            {active?.regions?.length > 0 && (
              <button className="btn-download" onClick={downloadHeightmap}>
                ↓ Descargar Heightmap PNG
              </button>
            )}
          </div>

          {/* ── Resize handle ── */}
          <div
            className="col-resizer"
            onMouseDown={(e) => {
              e.preventDefault()
              resizeDragRef.current = { active: true, startX: e.clientX, startWidth: rightWidth }
            }}
          />

          {/* ── Right column ── */}
          <div className="col-right" style={{ width: rightWidth }}>
            <DetectionPanel active={active} loading={loading} onDetect={detect}
              onPatch={(patch) => dispatch({ type: 'PATCH_ACTIVE', patch })} />

            {active?.regions?.length > 0 && (
              <RegionsPanel
                active={active} perfMs={perfMs} showNumbers={showNumbers}
                setShowNumbers={setShowNumbers}
                mergeMode={mergeMode} mergeSelection={mergeSelection}
                onToggleMergeMode={() => { setMergeMode(m => !m); setMergeSelection([]) }}
                onToggleMergeItem={toggleMergeItem}
                onDoMerge={doMerge}
                onCancelMerge={() => { setMergeMode(false); setMergeSelection([]) }}
                onPreset={applyPreset}
                onSetHeight={setHeight}
                onSetHeightWithUndo={setHeightWithUndo}
                onHover={(id) => {
                  hoverRef.current = id
                  const { level, panX, panY } = zoomRef.current
                  if (id != null) {
                    renderToCanvas(canvasRef.current, active, id)
                    drawLabelsOnOverlay(labelsCanvasRef.current, viewportRef.current, active, showNumRef.current, id, level, panX, panY, mergeSelRef.current)
                  } else {
                    renderView()
                  }
                }}
              />
            )}
          </div>
        </div>
      )}

      {toastMsg && <div className="toast" key={toastMsg + Date.now()}>{toastMsg}</div>}
    </div>
  )
}
