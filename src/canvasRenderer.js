import { regionColor } from './utils'

export function renderToCanvas(canvas, s, highlightId) {
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
