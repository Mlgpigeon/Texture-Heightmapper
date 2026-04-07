import { regionColor } from './utils'

export const LABEL_SIZE = 14 // fixed CSS-pixel font size — never scales with zoom

export function drawLabelOverlay(ctx, idx, cx, cy, selected) {
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
export function hitTestLabel(vp, s, showNumbers, zoomLevel, panX, panY, mx, my) {
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

export function drawLabelsOnOverlay(overlayCanvas, vp, s, showNumbers, highlightId, zoomLevel, panX, panY, mergeSelection) {
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
