import { PROCESSORS } from './constants'

export function makeImageState(filename, imgRGBA, width, height, thumbUrl) {
  const paramValues = {}
  PROCESSORS.connected.params.forEach(p => { paramValues[p.key] = p.default })
  return { filename, imgRGBA, imgWidth: width, imgHeight: height, thumbUrl,
    labelMap: null, regions: [], viewMode: 'original', processor: 'connected',
    paramValues, preBlur: 3, labelSmooth: 5, undoStack: [], redoStack: [] }
}

export const INIT = { images: [], activeIdx: -1 }

export function reducer(st, a) {
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
