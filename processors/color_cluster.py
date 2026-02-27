"""
Color Clustering processor.

Groups pixels purely by color similarity (ignoring spatial position).
Uses iterative centroid clustering similar to k-means but with
automatic cluster count based on tolerance.

Good for when you want all pixels of same color at same height
regardless of where they are.
"""

from __future__ import annotations
import numpy as np
from .base import BaseProcessor, Region, DetectionResult


class ColorClusterProcessor(BaseProcessor):
    name = "Clustering por Color"
    description = "Agrupa píxeles solo por color (ignora posición). Todos los píxeles del mismo color = misma región."

    def get_params(self) -> list[dict]:
        return [
            {
                "key": "tolerance",
                "label": "Tolerancia de color",
                "type": "slider",
                "min": 5, "max": 80, "step": 1,
                "default": 35,
                "hint": "Diferencia RGB máxima para agrupar colores"
            },
            {
                "key": "min_region_pct",
                "label": "Región mínima (%)",
                "type": "slider",
                "min": 0.1, "max": 5.0, "step": 0.1,
                "default": 0.5,
                "hint": "Descarta colores menores a este % de la imagen"
            },
            {
                "key": "max_samples",
                "label": "Muestras para detección",
                "type": "number",
                "min": 5000, "max": 100000, "step": 5000,
                "default": 30000,
                "hint": "Más muestras = más preciso pero más lento"
            },
        ]

    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        tolerance = params.get("tolerance", 35)
        min_pct = params.get("min_region_pct", 0.5)
        max_samples = int(params.get("max_samples", 30000))

        h, w = image.shape[:2]
        total = h * w
        rgb = image[:, :, :3].reshape(-1, 3).astype(np.float32)
        alpha = image[:, :, 3].reshape(-1)

        # Sample pixels for clustering
        opaque = np.where(alpha >= 128)[0]
        if len(opaque) == 0:
            return DetectionResult(
                label_map=np.full((h, w), -1, dtype=np.int32),
                regions=[], processor_name=self.name,
            )

        n_samples = min(max_samples, len(opaque))
        sample_idx = opaque[np.random.choice(len(opaque), n_samples, replace=False)]
        samples = rgb[sample_idx]

        # Greedy centroid clustering (on samples only — this is fast)
        centroids = []
        counts = []

        for px in samples:
            matched = False
            for i, c in enumerate(centroids):
                if np.sqrt(np.sum((px - c) ** 2)) < tolerance:
                    n = counts[i] + 1
                    centroids[i] = (centroids[i] * counts[i] + px) / n
                    counts[i] = n
                    matched = True
                    break
            if not matched:
                centroids.append(px.copy())
                counts.append(1)

        # Filter small clusters
        min_count = n_samples * (min_pct / 100)
        valid = [(c, n) for c, n in zip(centroids, counts) if n >= min_count]
        if not valid:
            valid = sorted(zip(centroids, counts), key=lambda x: -x[1])[:20]

        centroids_arr = np.array([c for c, _ in valid], dtype=np.float32)

        # VECTORIZED: Assign all opaque pixels to nearest centroid
        # Process in chunks to avoid memory explosion on 4K+ images
        label_map = np.full(total, -1, dtype=np.int32)
        chunk_size = 500_000  # Process 500k pixels at a time

        for start in range(0, len(opaque), chunk_size):
            end = min(start + chunk_size, len(opaque))
            chunk_idx = opaque[start:end]
            chunk_colors = rgb[chunk_idx]  # (chunk, 3)

            # Compute distances: (chunk, n_centroids)
            diffs = chunk_colors[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)  # Skip sqrt, argmin is the same
            label_map[chunk_idx] = np.argmin(dists, axis=1)

        # Build regions
        regions = []
        for i, (c, _) in enumerate(valid):
            mask = label_map == i
            count = int(np.sum(mask))
            if count == 0:
                continue

            positions = np.where(mask)[0]
            xs = positions % w
            ys = positions // w

            regions.append(Region(
                id=i,
                color=(int(c[0]), int(c[1]), int(c[2])),
                pixel_count=count,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            ))

        # Sort by pixel count and re-index
        regions.sort(key=lambda r: r.pixel_count, reverse=True)

        if regions:
            max_old_id = max(r.id for r in regions)
            id_remap = np.full(max_old_id + 1, -1, dtype=np.int32)
            for new_id, r in enumerate(regions):
                id_remap[r.id] = new_id
                r.id = new_id

            valid_mask = (label_map >= 0) & (label_map <= max_old_id)
            final_labels = np.full(total, -1, dtype=np.int32)
            final_labels[valid_mask] = id_remap[label_map[valid_mask]]
        else:
            final_labels = np.full(total, -1, dtype=np.int32)

        # Assign heights by luminance
        by_lum = sorted(regions, key=lambda r: 0.299*r.color[0] + 0.587*r.color[1] + 0.114*r.color[2])
        for i, r in enumerate(by_lum):
            r.height = int((i / max(1, len(by_lum) - 1)) * 255)

        return DetectionResult(
            label_map=final_labels.reshape(h, w),
            regions=regions,
            processor_name=self.name,
        )
