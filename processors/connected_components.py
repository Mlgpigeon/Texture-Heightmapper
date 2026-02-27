"""
Connected Component region detector.

Groups spatially connected pixels of similar color into independent regions.
Same color in different locations = different regions.

Algorithm:
  1. Fine quantization (tolerance//2) to create initial discrete color buckets.
  2. scipy.ndimage.label for fast connected component detection per bucket.
  3. Adjacency merge pass (Union-Find): adjacent regions whose average colors
     are within tolerance get merged. This fixes the hard bucket-boundary
     problem where two pixels with colors 29 and 31 (tolerance=30) would
     never connect under naive quantization.
  4. Small regions are absorbed into the nearest large region by color.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from .base import BaseProcessor, Region, DetectionResult


class ConnectedComponentProcessor(BaseProcessor):
    name = "Componentes Conectados"
    description = "Detecta regiones por color + proximidad espacial. Mismo color en zonas separadas = regiones distintas."

    def get_params(self) -> list[dict]:
        return [
            {
                "key": "tolerance",
                "label": "Tolerancia de color",
                "type": "slider",
                "min": 1, "max": 120, "step": 1,
                "default": 25,
                "hint": "Diferencia RGB máxima para considerar dos colores como la misma región"
            },
            {
                "key": "min_region_pct",
                "label": "Región mínima (%)",
                "type": "slider",
                "min": 0.01, "max": 10.0, "step": 0.01,
                "default": 0.3,
                "hint": "Descarta regiones menores a este % del total de píxeles"
            },
            {
                "key": "connectivity",
                "label": "Conectividad",
                "type": "select",
                "options": [
                    {"value": 4, "label": "4-vecinos (cruz)"},
                    {"value": 8, "label": "8-vecinos (incluye diagonales)"},
                ],
                "default": 4,
                "hint": "4 es más estricto, 8 conecta también por esquinas"
            },
            {
                "key": "merge_adjacent",
                "label": "Fusionar regiones adyacentes similares",
                "type": "checkbox",
                "default": True,
                "hint": "Fusiona regiones vecinas cuyos colores estén dentro de la tolerancia (corrige artefactos de borde)"
            },
        ]

    # ── Public detect ─────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        tolerance    = float(params.get("tolerance", 25))
        min_pct      = float(params.get("min_region_pct", 0.3))
        connectivity = int(params.get("connectivity", 4))
        merge_adj    = bool(params.get("merge_adjacent", True))

        h, w  = image.shape[:2]
        total = h * w
        rgb   = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3]
        opaque = alpha >= 128

        # ── Step 1: Fine quantization ────────────────────────────────────────
        # Use tolerance//2 so that neighboring color buckets still fall within
        # tolerance of each other — the merge pass below will reunite them.
        quant_step = max(1, int(tolerance) // 2)
        quantized  = (rgb / quant_step).astype(np.int32)
        # Collision-free encoding for uint8 values (max quantized = 255)
        color_id   = (quantized[:, :, 0] * 1_000_000
                      + quantized[:, :, 1] * 1_000
                      + quantized[:, :, 2])

        # ── Step 2: Structuring element ──────────────────────────────────────
        if connectivity == 8:
            struct = np.ones((3, 3), dtype=np.int32)
        else:
            struct = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.int32)

        # ── Step 3: Connected components per quantized color ─────────────────
        unique_colors = np.unique(color_id[opaque])
        label_map  = np.full((h, w), -1, dtype=np.int32)
        next_label = 0
        raw_regions: list[Region] = []

        for uc in unique_colors:
            mask = (color_id == uc) & opaque
            labeled, n_comp = ndimage.label(mask, structure=struct)

            for comp_id in range(1, n_comp + 1):
                comp_mask = labeled == comp_id
                count = int(np.sum(comp_mask))
                if count == 0:
                    continue

                label_map[comp_mask] = next_label
                avg  = rgb[comp_mask].mean(axis=0)
                ys, xs = np.where(comp_mask)
                raw_regions.append(Region(
                    id=next_label,
                    color=(int(avg[0]), int(avg[1]), int(avg[2])),
                    pixel_count=count,
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                ))
                next_label += 1

        if next_label == 0:
            return DetectionResult(
                label_map=label_map, regions=[], processor_name=self.name
            )

        # ── Step 4: Adjacency merge pass (Union-Find) ─────────────────────────
        # Merges neighboring regions whose average colors are within tolerance.
        # This is the key fix for the hard bucket-boundary problem.
        if merge_adj and len(raw_regions) > 1:
            label_map, raw_regions, next_label = self._merge_adjacent(
                label_map, raw_regions, next_label, tolerance
            )

        # ── Step 5: Filter small regions and absorb into nearest large ────────
        min_pixels = max(1, int(total * (min_pct / 100.0)))
        large = [r for r in raw_regions if r.pixel_count >= min_pixels]
        small = [r for r in raw_regions if r.pixel_count <  min_pixels]

        if large and small:
            small_ids    = np.array([r.id for r in small], dtype=np.int32)
            large_colors = np.array([r.color for r in large], dtype=np.float32)
            large_ids    = np.array([r.id   for r in large], dtype=np.int32)

            flat_labels = label_map.reshape(-1)
            flat_rgb    = rgb.reshape(-1, 3)
            is_small    = np.isin(flat_labels, small_ids)

            if np.any(is_small):
                chunk = 500_000
                small_idx = np.where(is_small)[0]
                for start in range(0, len(small_idx), chunk):
                    end  = min(start + chunk, len(small_idx))
                    cidx = small_idx[start:end]
                    diffs = flat_rgb[cidx][:, np.newaxis, :] - large_colors[np.newaxis, :, :]
                    flat_labels[cidx] = large_ids[np.argmin(np.sum(diffs ** 2, axis=2), axis=1)]
                label_map = flat_labels.reshape(h, w)

            for r in large:
                r.pixel_count = int(np.sum(label_map == r.id))

        # ── Step 6: Re-index sequentially by size ────────────────────────────
        pool   = large if large else raw_regions[:50]
        final  = sorted([r for r in pool if r.pixel_count > 0],
                        key=lambda r: r.pixel_count, reverse=True)

        if final:
            max_old = max(r.id for r in final)
            id_remap = np.full(max_old + 1, -1, dtype=np.int32)
            for new_id, r in enumerate(final):
                if r.id <= max_old:
                    id_remap[r.id] = new_id

            valid   = (label_map >= 0) & (label_map <= max_old)
            new_lm  = np.full((h, w), -1, dtype=np.int32)
            new_lm[valid] = id_remap[label_map[valid]]
            label_map = new_lm

            for i, r in enumerate(final):
                r.id = i

        # ── Step 7: Assign heights by luminance ──────────────────────────────
        by_lum = sorted(final,
                        key=lambda r: 0.299 * r.color[0]
                                    + 0.587 * r.color[1]
                                    + 0.114 * r.color[2])
        n = max(1, len(by_lum) - 1)
        for i, r in enumerate(by_lum):
            r.height = int((i / n) * 255)

        return DetectionResult(
            label_map=label_map,
            regions=final,
            processor_name=self.name,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _merge_adjacent(
        self,
        label_map: np.ndarray,
        regions: list[Region],
        next_label: int,
        tolerance: float,
    ) -> tuple[np.ndarray, list[Region], int]:
        """
        Union-Find merge of adjacent regions with similar average colors.

        For each pair of touching regions, if the Euclidean distance between
        their average colors is less than `tolerance`, they are merged into one.
        The merged region's color is the pixel-count-weighted average.
        """
        # Direct arrays indexed by region id (ids are 0..next_label-1)
        colors = np.zeros((next_label, 3), dtype=np.float64)
        counts = np.zeros(next_label,      dtype=np.float64)
        for r in regions:
            colors[r.id] = r.color
            counts[r.id] = r.pixel_count

        parent = np.arange(next_label, dtype=np.int32)

        # ── Find adjacent label pairs (vectorised) ────────────────────────────
        def _adj_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            diff = (a != b) & (a >= 0) & (b >= 0)
            if not diff.any():
                return np.empty((0, 2), dtype=np.int32)
            pts = np.column_stack([a[diff].ravel(), b[diff].ravel()])
            pts = np.sort(pts, axis=1)          # normalise so a ≤ b
            return np.unique(pts, axis=0)       # deduplicate

        pairs = np.vstack([
            _adj_pairs(label_map[:, :-1], label_map[:, 1:]),   # horizontal
            _adj_pairs(label_map[:-1, :], label_map[1:, :]),   # vertical
        ])
        pairs = np.unique(pairs, axis=0)

        # ── Union-Find helpers ────────────────────────────────────────────────
        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        tol_sq = tolerance * tolerance

        # ── Merge similar adjacent pairs ──────────────────────────────────────
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra == rb:
                continue
            dist_sq = float(np.sum((colors[ra] - colors[rb]) ** 2))
            if dist_sq < tol_sq:
                # Merge rb → ra with weighted-average color
                na, nb = counts[ra], counts[rb]
                total_n = na + nb
                if total_n > 0:
                    colors[ra] = (colors[ra] * na + colors[rb] * nb) / total_n
                counts[ra] = total_n
                parent[rb]  = ra

        # ── Compress all parent chains ─────────────────────────────────────────
        n_iters = int(np.log2(max(next_label, 2))) + 2
        for _ in range(n_iters):
            parent = parent[parent]

        # ── Remap label_map ───────────────────────────────────────────────────
        valid  = label_map >= 0
        new_lm = label_map.copy()
        new_lm[valid] = parent[label_map[valid]]

        # ── Rebuild region list ───────────────────────────────────────────────
        unique_ids = np.unique(new_lm[valid])
        h, w = new_lm.shape
        new_regions: list[Region] = []

        for uid in unique_ids:
            uid = int(uid)
            mask    = new_lm == uid
            count   = int(np.sum(mask))
            if count == 0:
                continue
            c = colors[uid]
            ys, xs = np.where(mask)
            new_regions.append(Region(
                id=uid,
                color=(int(np.clip(c[0], 0, 255)),
                       int(np.clip(c[1], 0, 255)),
                       int(np.clip(c[2], 0, 255))),
                pixel_count=count,
                bbox=(int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max())),
            ))

        new_next = int(max((r.id for r in new_regions), default=-1)) + 1
        return new_lm, new_regions, new_next
