import { useReducer, useRef, useEffect, useCallback, useState, memo } from 'react'
import { PROCESSORS } from './constants'
import { regionColor, typedArrayToBase64, base64ToInt32Array, imgRGBAtoDataURL } from './utils'

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════

function makeImageState(filename, imgRGBA, width, height, thumbUrl) {
  const paramValues = {}
  PROCESSORS.connected.params.forEach(p => { paramValues[p.key] = p.default })
  return { filename, imgRGBA, imgWidth: width, imgHeight: height, thumbUrl,
    labelMap: null, regions: [], viewMode: 'original', processor: 'connected',
    paramValues, preBlur: 3, labelSmooth: 5, undoStack: [], redoStack: [] }
}

const INIT = { images: [], activeIdx: -1 }

function reducer(st, a) {
  const { images: imgs, activeIdx: ai } = st
  const patchActive = patch => ({
    ...st, images: imgs.map((img, i) => i === ai ? { ...img, ...patch } : img)
  })

  switch (a.type) {
    case 'ADD_IMAGE': {
      const ni = [...imgs, a.img]
      return { images: ni, activeIdx: ni.length - 1 }
    }
    case 'REMOVE_IMAGE': {
      const ni = imgs.filter((_, i) => i !== a.idx)
      if (!ni.length) return INIT
      const na = ai === a.idx ? Math.min(a.idx, ni.length - 1)
        : ai > a.idx ? ai - 1 : ai
      return { images: ni, activeIdx: na }
    }
    case 'SET_ACTIVE': return { ...st, activeIdx: a.idx }
    case 'PATCH_ACTIVE': return patchActive(a.patch)
    case 'SET_HEIGHT': {
      const regions = imgs[ai].regions.map(r => r.id === a.id ? { ...r, height: a.h } : r)
      return patchActive({ regions })
    }
    case 'PUSH_UNDO': {
      const img = imgs[ai]
      // Store labelMap by reference — it's immutable (replaced, not mutated)
      const snap = { regions: JSON.parse(JSON.stringify(img.regions)),
        labelMap: img.labelMap }
      return patchActive({ undoStack: [...img.undoStack, snap].slice(-20), redoStack: [] })
    }
    case 'UNDO': {
      const img = imgs[ai]
      if (!img.undoStack.length) return st
      const current = { regions: JSON.parse(JSON.stringify(img.regions)),
        labelMap: img.labelMap }
      const snap = img.undoStack[img.undoStack.length - 1]
      return patchActive({ regions: snap.regions, labelMap: snap.labelMap,
        undoStack: img.undoStack.slice(0, -1),
        redoStack: [...img.redoStack, current].slice(-20) })
    }
    case 'REDO': {
      const img = imgs[ai]
      if (!img.redoStack.length) return st
      const current = { regions: JSON.parse(JSON.stringify(img.regions)),
        labelMap: img.labelMap }
      const snap = img.redoStack[img.redoStack.length - 1]
      return patchActive({ regions: snap.regions, labelMap: snap.labelMap,
        redoStack: img.redoStack.slice(0, -1),
        undoStack: [...img.undoStack, current].slice(-20) })
    }
    case 'APPLY_PRESET': {
      const img = imgs[ai]
      const mode = a.mode
      const sorted = [...img.regions].sort((a2, b2) => {
        const la = a2.color[0]*.299 + a2.color[1]*.587 + a2.color[2]*.114
        const lb = b2.color[0]*.299 + b2.color[1]*.587 + b2.color[2]*.114
        return la - lb
      })
      const rank = {}; sorted.forEach((r, i) => { rank[r.id] = i })
      const n = img.regions.length
      const regions = img.regions.map(r => {
        if (mode === 'flat') return { ...r, height: 128 }
        if (mode === 'by-area') return r // handled below
        const t = n > 1 ? rank[r.id] / (n - 1) : .5
        return { ...r, height: Math.round((mode === 'dark-high' ? 1 - t : t) * 255) }
      })
      if (mode === 'by-area') {
        const bySz = [...img.regions].sort((a2, b2) => b2.pixelCount - a2.pixelCount)
        const hMap = {}; bySz.forEach((r, i) => { hMap[r.id] = Math.round(i / Math.max(1, n-1) * 255) })
        return patchActive({ regions: img.regions.map(r => ({ ...r, height: hMap[r.id] })) })
      }
      return patchActive({ regions })
    }
    case 'MERGE': {
      const img = imgs[ai]
      const ids = a.ids.slice().sort((a2, b2) => a2 - b2)
      const keepId = ids[0]; const rem = new Set(ids.slice(1))
      const keep = img.regions.find(r => r.id === keepId)
      if (!keep) return st
      let px = keep.pixelCount, R = keep.color[0]*px, G = keep.color[1]*px, B = keep.color[2]*px
      for (const rid of rem) {
        const r = img.regions.find(r2 => r2.id === rid)
        if (!r) continue
        px += r.pixelCount; R += r.color[0]*r.pixelCount; G += r.color[1]*r.pixelCount; B += r.color[2]*r.pixelCount
      }
      const lm = new Int32Array(img.labelMap)
      for (let i = 0; i < lm.length; i++) if (rem.has(lm[i])) lm[i] = keepId
      const regions = img.regions.filter(r => !rem.has(r.id))
        .map(r => r.id === keepId ? { ...r, color: [(R/px)|0, (G/px)|0, (B/px)|0], pixelCount: px } : r)
      return patchActive({ regions, labelMap: lm })
    }
    case 'LOAD': return { images: a.images, activeIdx: a.activeIdx }
    default: return st
  }
}

// ═══════════════════════════════════════════════════════════════
// CANVAS RENDERING — main canvas (pixel data only, no labels)
// ═══════════════════════════════════════════════════════════════

function renderToCanvas(canvas, s, highlightId) {
  if (!canvas || !s) return
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  const { imgWidth: w, imgHeight: h } = s
  const n = w * h

  if (canvas.width !== w) canvas.width = w
  if (canvas.height !== h) canvas.height = h

  if (s.viewMode === 'original') {
    ctx.putImageData(new ImageData(new Uint8ClampedArray(s.imgRGBA), w, h), 0, 0)
    if (highlightId != null && s.labelMap) renderHighlight(ctx, s, highlightId, w, h)
    return
  }
  if (!s.labelMap || !s.regions.length) return

  const imgData = ctx.createImageData(w, h)
  const data = imgData.data
  const lm = s.labelMap
  const maxId = s.regions.reduce((m, r) => Math.max(m, r.id), 0)

  if (s.viewMode === 'regions') {
    const lutR = new Uint8Array(maxId + 1), lutG = new Uint8Array(maxId + 1), lutB = new Uint8Array(maxId + 1)
    s.regions.forEach((r, i) => { const c = regionColor(i); lutR[r.id]=c[0]; lutG[r.id]=c[1]; lutB[r.id]=c[2] })
    if (highlightId != null) {
      for (let i = 0; i < n; i++) {
        const p = i<<2; const lab = lm[i]; if (lab < 0) continue
        if (lab === highlightId) { data[p]=255; data[p+1]=0; data[p+2]=255; data[p+3]=255 }
        else { data[p]=40; data[p+1]=40; data[p+2]=40; data[p+3]=255 }
      }
    } else {
      for (let i = 0; i < n; i++) {
        const p = i<<2; const lab = lm[i]; if (lab < 0) continue
        data[p]=lutR[lab]; data[p+1]=lutG[lab]; data[p+2]=lutB[lab]; data[p+3]=255
      }
    }
  } else if (s.viewMode === 'heightmap') {
    const lutH = new Uint8Array(maxId + 1).fill(128)
    s.regions.forEach(r => { lutH[r.id] = r.height })
    for (let i = 0; i < n; i++) {
      const p = i<<2; const lab = lm[i]; if (lab < 0) continue
      const v = lutH[lab]; data[p]=v; data[p+1]=v; data[p+2]=v; data[p+3]=255
    }
  }
  ctx.putImageData(imgData, 0, 0)
  // Labels are drawn on the overlay canvas — never on the main canvas
}

function renderHighlight(ctx, s, highlightId, w, h) {
  const imgData = ctx.createImageData(w, h); const data = imgData.data; const lm = s.labelMap
  for (let i = 0; i < w*h; i++) {
    const p = i<<2; const lab = lm[i]; if (lab < 0) continue
    if (lab === highlightId) { data[p]=255; data[p+1]=0; data[p+2]=255; data[p+3]=255 }
    else { data[p]=40; data[p+1]=40; data[p+2]=40; data[p+3]=255 }
  }
  ctx.putImageData(imgData, 0, 0)
}

// ═══════════════════════════════════════════════════════════════
// LABELS OVERLAY — drawn at screen resolution, always crisp
// ═══════════════════════════════════════════════════════════════

const LABEL_SIZE = 14 // fixed CSS-pixel font size — never scales with zoom

function drawLabelOverlay(ctx, idx, cx, cy, selected) {
  const scale = LABEL_SIZE
  const text = '#' + idx
  ctx.font = `bold ${scale}px "Segoe UI", sans-serif`
  const tw = ctx.measureText(text).width; const pad = scale * .35
  const rx = cx - tw/2 - pad, ry = cy - scale/2 - pad, rw = tw + pad*2, rh = scale + pad*2, br = scale * .3
  if (selected) {
    ctx.fillStyle = 'rgba(0,180,100,0.92)'; ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, br); ctx.fill()
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2
    ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, br); ctx.stroke()
  } else {
    ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, br); ctx.fill()
    const c = regionColor(idx)
    ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, br); ctx.stroke()
  }
  ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, cx, cy)
}

/** Hit-test: returns the region index if (mx, my) in viewport-px falls on a label, else -1 */
function hitTestLabel(vp, s, showNumbers, zoomLevel, panX, panY, mx, my) {
  if (!vp || !s?.regions?.length) return -1
  if (s.viewMode === 'original' || !showNumbers) return -1
  const vw = vp.clientWidth, vh = vp.clientHeight
  for (let i = 0; i < s.regions.length; i++) {
    const r = s.regions[i]
    const cx = (r.bbox[0] + r.bbox[2]) / 2 * zoomLevel + panX
    const cy = (r.bbox[1] + r.bbox[3]) / 2 * zoomLevel + panY
    if (cx < -40 || cx > vw + 40 || cy < -20 || cy > vh + 20) continue
    const text = '#' + i
    const tw = text.length * (LABEL_SIZE * 0.62), pad = LABEL_SIZE * .35
    const rx = cx - tw / 2 - pad, ry = cy - LABEL_SIZE / 2 - pad
    if (mx >= rx && mx <= rx + tw + pad * 2 && my >= ry && my <= ry + LABEL_SIZE + pad * 2) return i
  }
  return -1
}

function drawLabelsOnOverlay(overlayCanvas, vp, s, showNumbers, highlightId, zoomLevel, panX, panY, mergeSelection) {
  if (!overlayCanvas || !vp) return
  const dpr = window.devicePixelRatio || 1
  const vw = vp.clientWidth, vh = vp.clientHeight
  if (!vw || !vh) return

  const cw = Math.round(vw * dpr), ch = Math.round(vh * dpr)
  if (overlayCanvas.width !== cw || overlayCanvas.height !== ch) {
    overlayCanvas.width = cw; overlayCanvas.height = ch
    overlayCanvas.style.width = vw + 'px'; overlayCanvas.style.height = vh + 'px'
  }

  const ctx = overlayCanvas.getContext('2d')
  ctx.clearRect(0, 0, cw, ch)

  if (!s?.regions?.length) return

  const selSet = mergeSelection?.length ? new Set(mergeSelection) : null

  ctx.save()
  ctx.scale(dpr, dpr)

  // Original view: only show label for the hovered region
  if (s.viewMode === 'original') {
    if (highlightId != null) {
      const idx = s.regions.findIndex(r => r.id === highlightId)
      if (idx >= 0) {
        const r = s.regions[idx]
        const cx = (r.bbox[0] + r.bbox[2]) / 2 * zoomLevel + panX
        const cy = (r.bbox[1] + r.bbox[3]) / 2 * zoomLevel + panY
        drawLabelOverlay(ctx, idx, cx, cy, selSet?.has(r.id))
      }
    }
    ctx.restore()
    return
  }

  // Regions / heightmap view
  if (!showNumbers) { ctx.restore(); return }

  if (highlightId != null) {
    // Only draw the highlighted region's label
    const idx = s.regions.findIndex(r => r.id === highlightId)
    if (idx >= 0) {
      const r = s.regions[idx]
      const cx = (r.bbox[0] + r.bbox[2]) / 2 * zoomLevel + panX
      const cy = (r.bbox[1] + r.bbox[3]) / 2 * zoomLevel + panY
      drawLabelOverlay(ctx, idx, cx, cy, selSet?.has(r.id))
    }
  } else {
    // Draw all labels that are within the viewport (with margin)
    s.regions.forEach((r, i) => {
      const cx = (r.bbox[0] + r.bbox[2]) / 2 * zoomLevel + panX
      const cy = (r.bbox[1] + r.bbox[3]) / 2 * zoomLevel + panY
      if (cx > -40 && cx < vw + 40 && cy > -20 && cy < vh + 20) {
        drawLabelOverlay(ctx, i, cx, cy, selSet?.has(r.id))
      }
    })
  }

  ctx.restore()
}

// ═══════════════════════════════════════════════════════════════
// APP
// ═══════════════════════════════════════════════════════════════

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
  const boxSelRef = useRef(null)       // { sx, sy, ex, ey } viewport-relative coords
  const boxElRef = useRef(null)        // DOM element for the selection rectangle

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
    // Update labels immediately so they follow pan/zoom without waiting for the debounced render
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
            {/* Image bar */}
            <ImageBar images={images} activeIdx={activeIdx}
              onSwitch={(i) => dispatch({ type: 'SET_ACTIVE', idx: i })}
              onRemove={(i) => dispatch({ type: 'REMOVE_IMAGE', idx: i })}
              onAdd={(files) => files.forEach(loadImageFile)} />

            {/* Canvas area */}
            <div className="canvas-area">
              {/* Tabs */}
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

              {/* Zoom bar */}
              <div className="zoom-bar">
                <button className="zoom-btn" onClick={() => adjustZoom(-.25)}>−</button>
                <span className="zoom-label">{zoomLabel}</span>
                <button className="zoom-btn" onClick={() => adjustZoom(.25)}>+</button>
                <button className="zoom-fit-btn" onClick={zoomFit}>Ajustar</button>
                <span style={{ fontSize: '11px', color: 'var(--text3)', marginLeft: '8px' }}>
                  Ctrl+Scroll · arrastrar
                </span>
              </div>

              {/* Viewport */}
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
                {/* Labels overlay — drawn at screen resolution, never blurry */}
                <canvas ref={labelsCanvasRef} className="labels-overlay" />
                <div ref={boxElRef} className="box-select" />
              </div>

              {/* Info bar */}
              <div className="canvas-info">
                <span>{active ? `${active.filename} — ${active.imgWidth}×${active.imgHeight}` : '—'}</span>
                <span>{active?.regions?.length > 0 ? `${active.regions.length} regiones` : ''}</span>
              </div>
            </div>

            {/* Download button */}
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

// ═══════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════

function UploadZone({ onFiles }) {
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

function ImageBar({ images, activeIdx, onSwitch, onRemove, onAdd }) {
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

function DetectionPanel({ active, loading, onDetect, onPatch }) {
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

function RegionsPanel({ active, perfMs, showNumbers, setShowNumbers, mergeMode, mergeSelection,
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
