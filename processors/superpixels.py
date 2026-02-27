"""
Superpixel (SLIC) region detector.

Uses scikit-image's SLIC algorithm for perceptually-aware segmentation.
Better at finding natural boundaries in textures with gradients.
Groups resulting superpixels by color similarity.
"""

from __future__ import annotations
import numpy as np
from .base import BaseProcessor, Region, DetectionResult


class SuperpixelProcessor(BaseProcessor):
    name = "Superpíxeles (SLIC)"
    description = "Segmentación perceptual avanzada. Mejor para texturas con gradientes. Agrupa superpíxeles similares."

    def get_params(self) -> list[dict]:
        return [
            {
                "key": "n_segments",
                "label": "Segmentos iniciales",
                "type": "slider",
                "min": 20, "max": 500, "step": 10,
                "default": 100,
                "hint": "Más = más fino, menos = más grueso"
            },
            {
                "key": "compactness",
                "label": "Compactación",
                "type": "slider",
                "min": 1, "max": 50, "step": 1,
                "default": 10,
                "hint": "Mayor = regiones más cuadradas, menor = sigue bordes de color"
            },
            {
                "key": "merge_tolerance",
                "label": "Tolerancia de fusión",
                "type": "slider",
                "min": 5, "max": 60, "step": 1,
                "default": 25,
                "hint": "Superpíxeles con color similar se fusionan"
            },
            {
                "key": "min_region_pct",
                "label": "Región mínima (%)",
                "type": "slider",
                "min": 0.1, "max": 5.0, "step": 0.1,
                "default": 0.5,
                "hint": "Descarta regiones finales menores a este %"
            },
        ]

    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        try:
            from skimage.segmentation import slic
        except ImportError:
            raise RuntimeError("scikit-image no instalado. Ejecuta: pip install scikit-image")

        n_segments = int(params.get("n_segments", 100))
        compactness = float(params.get("compactness", 10))
        merge_tol = float(params.get("merge_tolerance", 25))
        min_pct = float(params.get("min_region_pct", 0.5))

        h, w = image.shape[:2]
        total = h * w
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]

        # Run SLIC
        segments = slic(
            rgb, n_segments=n_segments, compactness=compactness,
            start_label=0, channel_axis=2,
        )

        # Compute average color per superpixel — vectorized with np.bincount
        n_sp = segments.max() + 1
        seg_flat = segments.reshape(-1)
        rgb_flat = rgb.reshape(-1, 3).astype(np.float64)

        sp_counts = np.bincount(seg_flat, minlength=n_sp)
        sp_colors = np.zeros((n_sp, 3), dtype=np.float64)
        for ch in range(3):
            sp_colors[:, ch] = np.bincount(seg_flat, weights=rgb_flat[:, ch], minlength=n_sp)

        nonzero = sp_counts > 0
        sp_colors[nonzero] /= sp_counts[nonzero, np.newaxis]

        # Build adjacency — vectorized using shifts
        adj = set()

        # Horizontal adjacency
        h_left = segments[:, :-1].ravel()
        h_right = segments[:, 1:].ravel()
        diff_h = h_left != h_right
        pairs_h = np.column_stack([h_left[diff_h], h_right[diff_h]])
        for a, b in pairs_h:
            adj.add((min(a, b), max(a, b)))

        # Vertical adjacency
        v_top = segments[:-1, :].ravel()
        v_bot = segments[1:, :].ravel()
        diff_v = v_top != v_bot
        pairs_v = np.column_stack([v_top[diff_v], v_bot[diff_v]])
        for a, b in pairs_v:
            adj.add((min(a, b), max(a, b)))

        # Union-find merge
        parent = np.arange(n_sp, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for a, b in adj:
            if np.sqrt(np.sum((sp_colors[a] - sp_colors[b]) ** 2)) < merge_tol:
                union(a, b)

        # Vectorized root finding for all superpixels
        # Flatten parent chain
        for _ in range(int(np.log2(n_sp)) + 2):
            parent = parent[parent]

        # Map roots to sequential IDs
        roots = parent[seg_flat]
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        root_to_id = np.arange(len(unique_roots), dtype=np.int32)

        # Build label map
        alpha_flat = alpha.reshape(-1)
        opaque = alpha_flat >= 128
        label_map = np.full(total, -1, dtype=np.int32)
        label_map[opaque] = root_to_id[inverse[opaque]]

        n_regions = len(unique_roots)

        # Compute region stats — vectorized
        valid_labels = label_map[opaque]
        valid_rgb = rgb_flat[opaque]

        region_counts = np.bincount(valid_labels, minlength=n_regions)
        region_colors = np.zeros((n_regions, 3), dtype=np.float64)
        for ch in range(3):
            region_colors[:, ch] = np.bincount(valid_labels, weights=valid_rgb[:, ch], minlength=n_regions)

        nonzero_r = region_counts > 0
        region_colors[nonzero_r] /= region_counts[nonzero_r, np.newaxis]

        # Compute bounding boxes — vectorized
        valid_indices = np.where(opaque)[0]
        valid_xs = valid_indices % w
        valid_ys = valid_indices // w

        region_bbox = np.zeros((n_regions, 4), dtype=np.int32)
        region_bbox[:, 0] = w   # min_x init
        region_bbox[:, 1] = h   # min_y init

        for rid in range(n_regions):
            if region_counts[rid] == 0:
                continue
            mask = valid_labels == rid
            rxs = valid_xs[mask]
            rys = valid_ys[mask]
            region_bbox[rid] = [rxs.min(), rys.min(), rxs.max(), rys.max()]

        # Build region objects, filter small
        min_pixels = int(total * (min_pct / 100))
        regions = []
        for i in range(n_regions):
            if region_counts[i] < min_pixels:
                continue
            c = region_colors[i]
            bb = region_bbox[i]
            regions.append(Region(
                id=i,
                color=(int(c[0]), int(c[1]), int(c[2])),
                pixel_count=int(region_counts[i]),
                bbox=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
            ))

        # Merge tiny regions into nearest large — vectorized
        large_ids = set(r.id for r in regions)
        if large_ids and len(large_ids) < n_regions:
            large_colors = np.array([r.color for r in regions], dtype=np.float32)
            large_id_arr = np.array([r.id for r in regions], dtype=np.int32)

            # Find pixels in small regions
            is_small = opaque.copy()
            for lid in large_ids:
                is_small &= (label_map.ravel() != lid)
            is_small &= (label_map.ravel() >= 0)

            if np.any(is_small):
                small_colors = rgb_flat[is_small]
                # Chunked distance computation
                chunk_size = 500_000
                small_indices = np.where(is_small)[0]
                for start in range(0, len(small_indices), chunk_size):
                    end = min(start + chunk_size, len(small_indices))
                    chunk_idx = small_indices[start:end]
                    chunk_colors = rgb_flat[chunk_idx].astype(np.float32)
                    diffs = chunk_colors[:, np.newaxis, :] - large_colors[np.newaxis, :, :]
                    dists = np.sum(diffs ** 2, axis=2)
                    label_map.ravel()[chunk_idx] = large_id_arr[np.argmin(dists, axis=1)]

        # Re-index
        regions.sort(key=lambda r: r.pixel_count, reverse=True)
        if regions:
            max_old_id = max(r.id for r in regions)
            old_to_new = np.full(max_old_id + 1, -1, dtype=np.int32)
            for new_id, r in enumerate(regions):
                old_to_new[r.id] = new_id
                r.id = new_id

            flat = label_map.ravel()
            valid_mask = (flat >= 0) & (flat <= max_old_id)
            final_labels = np.full(total, -1, dtype=np.int32)
            final_labels[valid_mask] = old_to_new[flat[valid_mask]]
        else:
            final_labels = np.full(total, -1, dtype=np.int32)

        # Heights by luminance
        by_lum = sorted(regions, key=lambda r: 0.299*r.color[0] + 0.587*r.color[1] + 0.114*r.color[2])
        for i, r in enumerate(by_lum):
            r.height = int((i / max(1, len(by_lum) - 1)) * 255)

        return DetectionResult(
            label_map=final_labels.reshape(h, w),
            regions=regions,
            processor_name=self.name,
            metadata={"n_superpixels_raw": n_sp, "n_merged": len(regions)},
        )
