'use strict';
// ═══════════════════════════════════════════════════════════════
// Heightmap Studio — Detection Web Worker
// All CPU-heavy detection runs here, off the main thread.
// ═══════════════════════════════════════════════════════════════

let W = 0, H = 0, N = 0;
let rgba = null; // Uint8Array (original image RGBA)

onmessage = function(e) {
  const msg = e.data;
  if (msg.type !== 'detect') return;

  W = msg.width;
  H = msg.height;
  N = W * H;
  rgba = new Uint8Array(msg.rgba);

  const params = msg.params;
  const t0 = performance.now();

  try {
    // ── Pre-process: box blur ──
    let src = rgba;
    const preBlur = params._pre_blur | 0;
    if (preBlur > 0) {
      src = boxBlurRGBA(rgba, W, H, preBlur);
    }

    // ── Detect ──
    let labelMap, regions;
    if (msg.processor === 'connected') {
      ({ labelMap, regions } = detectCC(src, W, H, params));
    } else {
      ({ labelMap, regions } = detectCluster(src, W, H, params));
    }

    // ── Post-process: mode filter on labels ──
    const labelSmooth = params._label_smooth | 0;
    if (labelSmooth > 0) {
      labelMap = modeFilter(labelMap, W, H, labelSmooth);
      recount(labelMap, rgba, W, H, regions);
    }

    // ── Assign heights by luminance ──
    assignHeights(regions);

    const elapsed = performance.now() - t0;

    postMessage({
      type: 'result',
      regions,
      labelMap: labelMap.buffer,
      elapsed: Math.round(elapsed),
    }, [labelMap.buffer]);

  } catch (err) {
    postMessage({ type: 'error', message: err.message });
  }
};

// ═══════════════════════════════════════════════════════════════
// CONNECTED COMPONENTS — Union-Find (single-pass, O(n·α(n)))
// ═══════════════════════════════════════════════════════════════

function detectCC(src, w, h, params) {
  const tol = params.tolerance || 30;
  const minPx = params.min_region_px || 50;
  const conn8 = (params.connectivity | 0) === 8;
  const n = w * h;

  // Quantize colors → uint32 color ID
  const step = Math.max(1, tol);
  const colorId = new Uint32Array(n);
  const opaque = new Uint8Array(n); // 1 = opaque

  for (let i = 0; i < n; i++) {
    const p = i << 2;
    if (src[p + 3] >= 128) {
      opaque[i] = 1;
      const qr = (src[p] / step) | 0;
      const qg = (src[p + 1] / step) | 0;
      const qb = (src[p + 2] / step) | 0;
      colorId[i] = qr * 65536 + qg * 256 + qb;
    }
  }

  // Union-Find
  const parent = new Int32Array(n);
  const rank = new Uint8Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;

  // Scan image and union adjacent pixels with same quantized color
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      if (!opaque[i]) continue;
      const cid = colorId[i];

      // Right neighbor
      if (x + 1 < w && opaque[i + 1] && colorId[i + 1] === cid) {
        union(parent, rank, i, i + 1);
      }
      // Bottom neighbor
      if (y + 1 < h) {
        const below = i + w;
        if (opaque[below] && colorId[below] === cid) {
          union(parent, rank, i, below);
        }
        // Diagonals (8-connectivity)
        if (conn8) {
          if (x + 1 < w && opaque[below + 1] && colorId[below + 1] === cid) {
            union(parent, rank, i, below + 1);
          }
          if (x > 0 && opaque[below - 1] && colorId[below - 1] === cid) {
            union(parent, rank, i, below - 1);
          }
        }
      }
    }
  }

  // Flatten parents (full path compression)
  for (let i = 0; i < n; i++) {
    if (opaque[i]) parent[i] = find(parent, i);
  }

  // Map roots → dense IDs
  const rootId = new Int32Array(n).fill(-1);
  let nextId = 0;
  for (let i = 0; i < n; i++) {
    if (!opaque[i]) continue;
    const root = parent[i];
    if (rootId[root] === -1) rootId[root] = nextId++;
  }

  const nReg = nextId;
  const labelMap = new Int32Array(n).fill(-1);
  const counts = new Int32Array(nReg);
  const sumR = new Float64Array(nReg);
  const sumG = new Float64Array(nReg);
  const sumB = new Float64Array(nReg);
  const minX = new Int32Array(nReg).fill(w);
  const minY = new Int32Array(nReg).fill(h);
  const maxX = new Int32Array(nReg);
  const maxY = new Int32Array(nReg);

  for (let i = 0; i < n; i++) {
    if (!opaque[i]) continue;
    const id = rootId[parent[i]];
    labelMap[i] = id;
    counts[id]++;
    const p = i << 2;
    sumR[id] += rgba[p];     // Use original colors, not blurred
    sumG[id] += rgba[p + 1];
    sumB[id] += rgba[p + 2];
    const x = i % w, y = (i / w) | 0;
    if (x < minX[id]) minX[id] = x;
    if (y < minY[id]) minY[id] = y;
    if (x > maxX[id]) maxX[id] = x;
    if (y > maxY[id]) maxY[id] = y;
  }

  // Build region objects
  const minPixels = minPx | 0;
  const large = [];
  const smallIds = new Set();

  for (let id = 0; id < nReg; id++) {
    if (counts[id] >= minPixels) {
      large.push({
        id,
        color: [
          (sumR[id] / counts[id]) | 0,
          (sumG[id] / counts[id]) | 0,
          (sumB[id] / counts[id]) | 0,
        ],
        pixelCount: counts[id],
        bbox: [minX[id], minY[id], maxX[id], maxY[id]],
        height: 128,
      });
    } else {
      smallIds.add(id);
    }
  }

  // Reassign small regions to nearest large region (color LUT)
  if (large.length > 0 && smallIds.size > 0) {
    reassignSmall(labelMap, rgba, w, h, n, large, smallIds);
  }

  // Re-index sequentially by pixel count (descending)
  large.sort((a, b) => b.pixelCount - a.pixelCount);
  const oldToNew = new Int32Array(nReg).fill(-1);
  large.forEach((r, i) => {
    oldToNew[r.id] = i;
    r.id = i;
  });

  for (let i = 0; i < n; i++) {
    if (labelMap[i] >= 0) {
      labelMap[i] = oldToNew[labelMap[i]];
    }
  }

  // Recount after reassignment
  for (const r of large) {
    r.pixelCount = 0;
    r.bbox = [w, h, 0, 0];
  }
  for (let i = 0; i < n; i++) {
    const id = labelMap[i];
    if (id < 0) continue;
    const r = large[id];
    r.pixelCount++;
    const x = i % w, y = (i / w) | 0;
    if (x < r.bbox[0]) r.bbox[0] = x;
    if (y < r.bbox[1]) r.bbox[1] = y;
    if (x > r.bbox[2]) r.bbox[2] = x;
    if (y > r.bbox[3]) r.bbox[3] = y;
  }

  return { labelMap, regions: large.filter(r => r.pixelCount > 0) };
}

// ═══════════════════════════════════════════════════════════════
// COLOR CLUSTERING — Greedy centroids + vectorized assignment
// ═══════════════════════════════════════════════════════════════

function detectCluster(src, w, h, params) {
  const tol = params.tolerance || 35;
  const minPx = params.min_region_px || 50;
  const maxSamples = params.max_samples || 30000;
  const n = w * h;
  const tolSq = tol * tol;

  // Collect opaque pixel indices
  const opaqueIdx = [];
  for (let i = 0; i < n; i++) {
    if (src[(i << 2) + 3] >= 128) opaqueIdx.push(i);
  }

  if (opaqueIdx.length === 0) {
    return { labelMap: new Int32Array(n).fill(-1), regions: [] };
  }

  // Fisher-Yates partial shuffle for sampling
  const nSamples = Math.min(maxSamples, opaqueIdx.length);
  for (let i = 0; i < nSamples; i++) {
    const j = i + (Math.random() * (opaqueIdx.length - i)) | 0;
    const tmp = opaqueIdx[i]; opaqueIdx[i] = opaqueIdx[j]; opaqueIdx[j] = tmp;
  }

  // Greedy centroid clustering on samples
  const centroids = []; // [[r,g,b], ...]
  const centCounts = [];

  for (let s = 0; s < nSamples; s++) {
    const idx = opaqueIdx[s];
    const p = idx << 2;
    const cr = src[p], cg = src[p + 1], cb = src[p + 2];

    let matched = -1;
    let bestDist = tolSq;
    for (let c = 0; c < centroids.length; c++) {
      const dr = cr - centroids[c][0];
      const dg = cg - centroids[c][1];
      const db = cb - centroids[c][2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestDist) { bestDist = d; matched = c; }
    }

    if (matched >= 0) {
      const cnt = centCounts[matched] + 1;
      const cc = centroids[matched];
      cc[0] = (cc[0] * centCounts[matched] + cr) / cnt;
      cc[1] = (cc[1] * centCounts[matched] + cg) / cnt;
      cc[2] = (cc[2] * centCounts[matched] + cb) / cnt;
      centCounts[matched] = cnt;
    } else {
      centroids.push([cr, cg, cb]);
      centCounts.push(1);
    }
  }

  // Filter small clusters (scale pixel threshold proportionally to sample size)
  const minCount = Math.round(minPx * nSamples / n);
  let valid = [];
  for (let i = 0; i < centroids.length; i++) {
    if (centCounts[i] >= minCount) valid.push(i);
  }
  if (valid.length === 0) {
    // Take top 20 by count
    const indexed = centroids.map((c, i) => [i, centCounts[i]]);
    indexed.sort((a, b) => b[1] - a[1]);
    valid = indexed.slice(0, 20).map(x => x[0]);
  }

  // Build centroid lookup arrays
  const nc = valid.length;
  const cR = new Float32Array(nc);
  const cG = new Float32Array(nc);
  const cB = new Float32Array(nc);
  for (let i = 0; i < nc; i++) {
    const c = centroids[valid[i]];
    cR[i] = c[0]; cG[i] = c[1]; cB[i] = c[2];
  }

  // Assign all opaque pixels to nearest centroid
  const labelMap = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const p = i << 2;
    if (src[p + 3] < 128) continue;
    const r = src[p], g = src[p + 1], b = src[p + 2];

    let bestDist = Infinity, bestId = 0;
    for (let c = 0; c < nc; c++) {
      const dr = r - cR[c], dg = g - cG[c], db = b - cB[c];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestDist) { bestDist = d; bestId = c; }
    }
    labelMap[i] = bestId;
  }

  // Build regions
  const counts = new Int32Array(nc);
  const sumR = new Float64Array(nc);
  const sumG = new Float64Array(nc);
  const sumB = new Float64Array(nc);
  const bboxMinX = new Int32Array(nc).fill(w);
  const bboxMinY = new Int32Array(nc).fill(h);
  const bboxMaxX = new Int32Array(nc);
  const bboxMaxY = new Int32Array(nc);

  for (let i = 0; i < n; i++) {
    const id = labelMap[i];
    if (id < 0) continue;
    counts[id]++;
    const p = i << 2;
    sumR[id] += rgba[p];
    sumG[id] += rgba[p + 1];
    sumB[id] += rgba[p + 2];
    const x = i % w, y = (i / w) | 0;
    if (x < bboxMinX[id]) bboxMinX[id] = x;
    if (y < bboxMinY[id]) bboxMinY[id] = y;
    if (x > bboxMaxX[id]) bboxMaxX[id] = x;
    if (y > bboxMaxY[id]) bboxMaxY[id] = y;
  }

  const minPixels = minPx | 0;
  const regions = [];
  for (let i = 0; i < nc; i++) {
    if (counts[i] < minPixels) continue;
    regions.push({
      id: i,
      color: [(sumR[i] / counts[i]) | 0, (sumG[i] / counts[i]) | 0, (sumB[i] / counts[i]) | 0],
      pixelCount: counts[i],
      bbox: [bboxMinX[i], bboxMinY[i], bboxMaxX[i], bboxMaxY[i]],
      height: 128,
    });
  }

  // Reassign small clusters to nearest large BY COLOR (not spatial —
  // color clustering intentionally ignores position)
  const largeIds = new Set(regions.map(r => r.id));
  if (largeIds.size > 0 && largeIds.size < nc) {
    const smallSet = new Set();
    for (let i = 0; i < nc; i++) {
      if (!largeIds.has(i)) smallSet.add(i);
    }
    reassignSmallByColor(labelMap, rgba, n, regions, smallSet);
  }

  // Re-index
  regions.sort((a, b) => b.pixelCount - a.pixelCount);
  const remap = new Int32Array(nc).fill(-1);
  regions.forEach((r, i) => { remap[r.id] = i; r.id = i; });
  for (let i = 0; i < n; i++) {
    if (labelMap[i] >= 0) labelMap[i] = remap[labelMap[i]];
  }

  // Recount
  for (const r of regions) { r.pixelCount = 0; r.bbox = [w, h, 0, 0]; }
  for (let i = 0; i < n; i++) {
    const id = labelMap[i];
    if (id < 0) continue;
    regions[id].pixelCount++;
    const x = i % w, y = (i / w) | 0;
    if (x < regions[id].bbox[0]) regions[id].bbox[0] = x;
    if (y < regions[id].bbox[1]) regions[id].bbox[1] = y;
    if (x > regions[id].bbox[2]) regions[id].bbox[2] = x;
    if (y > regions[id].bbox[3]) regions[id].bbox[3] = y;
  }

  return { labelMap, regions: regions.filter(r => r.pixelCount > 0) };
}

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

function find(parent, x) {
  while (parent[x] !== x) {
    parent[x] = parent[parent[x]]; // path halving
    x = parent[x];
  }
  return x;
}

function union(parent, rank, a, b) {
  a = find(parent, a);
  b = find(parent, b);
  if (a === b) return;
  if (rank[a] < rank[b]) { const t = a; a = b; b = t; }
  parent[b] = a;
  if (rank[a] === rank[b]) rank[a]++;
}

// Reassign pixels in small regions to nearest large region using color LUT
// Reassign small regions to nearest SPATIALLY ADJACENT large region.
// Uses iterative dilation: each pass, small-region pixels touching a large
// region get absorbed into it. Repeats until all are assigned.
// This preserves connected-component semantics (no merging across gaps).
function reassignSmall(labelMap, rgba, w, h, n, largeRegions, smallIds) {
  const largeSet = new Set(largeRegions.map(r => r.id));
  let remaining = 0;

  // Count how many small pixels we need to reassign
  for (let i = 0; i < n; i++) {
    if (labelMap[i] >= 0 && smallIds.has(labelMap[i])) remaining++;
  }

  // Iterative dilation — typically converges in a few passes
  const maxPasses = 200;
  for (let pass = 0; pass < maxPasses && remaining > 0; pass++) {
    let changed = 0;

    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (labelMap[i] < 0 || !smallIds.has(labelMap[i])) continue;

        // Check 4-connected neighbors for a large region
        let bestId = -1;

        if (x > 0 && labelMap[i - 1] >= 0 && largeSet.has(labelMap[i - 1])) bestId = labelMap[i - 1];
        else if (x + 1 < w && labelMap[i + 1] >= 0 && largeSet.has(labelMap[i + 1])) bestId = labelMap[i + 1];
        else if (y > 0 && labelMap[i - w] >= 0 && largeSet.has(labelMap[i - w])) bestId = labelMap[i - w];
        else if (y + 1 < h && labelMap[i + w] >= 0 && largeSet.has(labelMap[i + w])) bestId = labelMap[i + w];

        if (bestId >= 0) {
          labelMap[i] = bestId;
          changed++;
        }
      }
    }

    remaining -= changed;
    if (changed === 0) break; // No more pixels can be reached — isolated islands
  }

  // Any remaining isolated small regions that couldn't reach a large region
  // via adjacency: just mark them as unassigned (-1) so they don't corrupt
  // the connected-component semantics.
  if (remaining > 0) {
    for (let i = 0; i < n; i++) {
      if (labelMap[i] >= 0 && smallIds.has(labelMap[i])) {
        labelMap[i] = -1;
      }
    }
  }
}

// Reassign small regions by COLOR distance (for color clustering mode,
// where spatial position is intentionally ignored).
function reassignSmallByColor(labelMap, rgba, n, largeRegions, smallIds) {
  // Build quantized color → large region LUT
  const levels = 64;
  const levels2 = levels * levels;
  const levels3 = levels2 * levels;
  const lut = new Int32Array(levels3).fill(-1);

  for (let ri = 0; ri < levels; ri++) {
    const rc = ri * 4 + 2;
    for (let gi = 0; gi < levels; gi++) {
      const gc = gi * 4 + 2;
      for (let bi = 0; bi < levels; bi++) {
        const bc = bi * 4 + 2;
        let bestDist = Infinity, bestId = largeRegions[0].id;
        for (const r of largeRegions) {
          const dr = rc - r.color[0], dg = gc - r.color[1], db = bc - r.color[2];
          const d = dr * dr + dg * dg + db * db;
          if (d < bestDist) { bestDist = d; bestId = r.id; }
        }
        lut[ri * levels2 + gi * levels + bi] = bestId;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (labelMap[i] >= 0 && smallIds.has(labelMap[i])) {
      const p = i << 2;
      labelMap[i] = lut[(rgba[p] >> 2) * levels2 + (rgba[p+1] >> 2) * levels + (rgba[p+2] >> 2)];
    }
  }
}

// Fast separable box blur (3 passes ≈ Gaussian)
function boxBlurRGBA(src, w, h, radius) {
  const n = w * h;
  const dst = new Uint8Array(n * 4);
  dst.set(src);

  const passes = 3;
  for (let p = 0; p < passes; p++) {
    boxBlurH(dst, w, h, radius);
    boxBlurV(dst, w, h, radius);
  }
  return dst;
}

function boxBlurH(data, w, h, r) {
  const d = 2 * r + 1;
  const inv = 1 / d;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sr = 0, sg = 0, sb = 0;
    // Initialize window
    for (let x = -r; x <= r; x++) {
      const cx = Math.max(0, Math.min(w - 1, x));
      const p = (row + cx) << 2;
      sr += data[p]; sg += data[p + 1]; sb += data[p + 2];
    }
    for (let x = 0; x < w; x++) {
      const p = (row + x) << 2;
      data[p] = (sr * inv + 0.5) | 0;
      data[p + 1] = (sg * inv + 0.5) | 0;
      data[p + 2] = (sb * inv + 0.5) | 0;
      // Slide window
      const addX = Math.min(w - 1, x + r + 1);
      const remX = Math.max(0, x - r);
      const pa = (row + addX) << 2;
      const pr = (row + remX) << 2;
      sr += data[pa] - data[pr];
      sg += data[pa + 1] - data[pr + 1];
      sb += data[pa + 2] - data[pr + 2];
    }
  }
}

function boxBlurV(data, w, h, r) {
  const d = 2 * r + 1;
  const inv = 1 / d;
  for (let x = 0; x < w; x++) {
    let sr = 0, sg = 0, sb = 0;
    for (let y = -r; y <= r; y++) {
      const cy = Math.max(0, Math.min(h - 1, y));
      const p = (cy * w + x) << 2;
      sr += data[p]; sg += data[p + 1]; sb += data[p + 2];
    }
    for (let y = 0; y < h; y++) {
      const p = (y * w + x) << 2;
      data[p] = (sr * inv + 0.5) | 0;
      data[p + 1] = (sg * inv + 0.5) | 0;
      data[p + 2] = (sb * inv + 0.5) | 0;
      const addY = Math.min(h - 1, y + r + 1);
      const remY = Math.max(0, y - r);
      const pa = (addY * w + x) << 2;
      const pr = (remY * w + x) << 2;
      sr += data[pa] - data[pr];
      sg += data[pa + 1] - data[pr + 1];
      sb += data[pa + 2] - data[pr + 2];
    }
  }
}

// Mode filter on label map (replaces thin-line artifacts)
function modeFilter(labelMap, w, h, kernelSize) {
  const half = (kernelSize >> 1) || 1;
  const n = w * h;
  const out = new Int32Array(n);
  const votes = new Map();

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (labelMap[i] < 0) { out[i] = -1; continue; }

      votes.clear();
      let maxCount = 0, maxLabel = labelMap[i];

      const y0 = Math.max(0, y - half), y1 = Math.min(h - 1, y + half);
      const x0 = Math.max(0, x - half), x1 = Math.min(w - 1, x + half);

      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          const lab = labelMap[yy * w + xx];
          if (lab < 0) continue;
          const c = (votes.get(lab) || 0) + 1;
          votes.set(lab, c);
          if (c > maxCount) { maxCount = c; maxLabel = lab; }
        }
      }
      out[i] = maxLabel;
    }
  }
  return out;
}

function recount(labelMap, rgba, w, h, regions) {
  const n = w * h;
  for (const r of regions) {
    r.pixelCount = 0;
    r.bbox = [w, h, 0, 0];
    r.color = [0, 0, 0];
  }
  const sums = regions.map(() => [0, 0, 0]);

  for (let i = 0; i < n; i++) {
    const id = labelMap[i];
    if (id < 0 || id >= regions.length) continue;
    const r = regions[id];
    if (!r) continue;
    r.pixelCount++;
    const p = i << 2;
    sums[id][0] += rgba[p];
    sums[id][1] += rgba[p + 1];
    sums[id][2] += rgba[p + 2];
    const x = i % w, y = (i / w) | 0;
    if (x < r.bbox[0]) r.bbox[0] = x;
    if (y < r.bbox[1]) r.bbox[1] = y;
    if (x > r.bbox[2]) r.bbox[2] = x;
    if (y > r.bbox[3]) r.bbox[3] = y;
  }

  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    if (r.pixelCount > 0) {
      r.color = [
        (sums[i][0] / r.pixelCount) | 0,
        (sums[i][1] / r.pixelCount) | 0,
        (sums[i][2] / r.pixelCount) | 0,
      ];
    }
  }

  // Remove empty
  for (let i = regions.length - 1; i >= 0; i--) {
    if (regions[i].pixelCount === 0) regions.splice(i, 1);
  }
}

function assignHeights(regions) {
  const sorted = [...regions].sort((a, b) => {
    const la = a.color[0] * 0.299 + a.color[1] * 0.587 + a.color[2] * 0.114;
    const lb = b.color[0] * 0.299 + b.color[1] * 0.587 + b.color[2] * 0.114;
    return la - lb;
  });
  const n = sorted.length;
  sorted.forEach((r, i) => {
    r.height = Math.round((i / Math.max(1, n - 1)) * 255);
  });
}
