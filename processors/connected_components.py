"""
Connected Component region detector.

Groups spatially connected pixels of similar color into independent regions.
Same color in different locations = different regions.

Uses quantization + scipy.ndimage.label for fast vectorized detection.
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
                "min": 5, "max": 80, "step": 1,
                "default": 30,
                "hint": "Diferencia RGB máxima dentro de una región"
            },
            {
                "key": "min_region_pct",
                "label": "Región mínima (%)",
                "type": "slider",
                "min": 0.05, "max": 5.0, "step": 0.05,
                "default": 0.5,
                "hint": "Descarta regiones menores a este % de la imagen"
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
                "hint": "4 es más estricto, 8 conecta más"
            },
        ]

    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        tolerance = params.get("tolerance", 30)
        min_pct = params.get("min_region_pct", 0.5)
        connectivity = int(params.get("connectivity", 4))

        h, w = image.shape[:2]
        total = h * w
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3]

        # Quantize colors by tolerance to create a discrete color map
        # This groups similar colors together before connected component analysis
        quant_step = max(1, tolerance)
        quantized = (rgb / quant_step).astype(np.int32)
        # Encode quantized RGB into a single integer per pixel
        color_id = quantized[:, :, 0] * 1000000 + quantized[:, :, 1] * 1000 + quantized[:, :, 2]

        # Mask out transparent pixels
        opaque = alpha >= 128

        # Build structuring element for connectivity
        if connectivity == 8:
            struct = np.ones((3, 3), dtype=np.int32)
        else:
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

        # Find unique quantized colors and run connected components per color
        unique_colors = np.unique(color_id[opaque])
        label_map = np.full((h, w), -1, dtype=np.int32)
        next_label = 0
        raw_regions = []

        for uc in unique_colors:
            # Mask of pixels with this quantized color that are opaque
            mask = (color_id == uc) & opaque

            # Label connected components within this color
            labeled, n_components = ndimage.label(mask, structure=struct)

            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id
                count = int(np.sum(comp_mask))
                if count == 0:
                    continue

                # Assign label
                label_map[comp_mask] = next_label

                # Compute average color
                comp_pixels = rgb[comp_mask]
                avg_color = comp_pixels.mean(axis=0)

                # Compute bounding box
                ys, xs = np.where(comp_mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                raw_regions.append(Region(
                    id=next_label,
                    color=(int(avg_color[0]), int(avg_color[1]), int(avg_color[2])),
                    pixel_count=count,
                    bbox=bbox,
                ))
                next_label += 1

        # Filter small regions
        min_pixels = int(total * (min_pct / 100))
        large = [r for r in raw_regions if r.pixel_count >= min_pixels]
        small = [r for r in raw_regions if r.pixel_count < min_pixels]

        if large and small:
            small_ids = np.array([r.id for r in small], dtype=np.int32)
            large_colors = np.array([r.color for r in large], dtype=np.float32)
            large_ids = np.array([r.id for r in large], dtype=np.int32)

            # Vectorized: find all pixels belonging to small regions
            small_id_set = set(small_ids.tolist())
            flat_labels = label_map.reshape(-1)
            flat_rgb = rgb.reshape(-1, 3)

            # Create mask of pixels in small regions
            is_small = np.isin(flat_labels, small_ids)
            if np.any(is_small):
                small_pixel_colors = flat_rgb[is_small]
                # Compute distances to all large region colors at once
                # shape: (n_small_pixels, n_large_regions)
                diffs = small_pixel_colors[:, np.newaxis, :] - large_colors[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs ** 2, axis=2))
                nearest = np.argmin(dists, axis=1)
                flat_labels[is_small] = large_ids[nearest]
                label_map = flat_labels.reshape(h, w)

            # Recount
            for r in large:
                r.pixel_count = int(np.sum(label_map == r.id))

        # Re-index sequentially
        final = sorted(
            [r for r in (large if large else raw_regions[:50]) if r.pixel_count > 0],
            key=lambda r: r.pixel_count,
            reverse=True,
        )

        if final:
            old_ids = np.array([r.id for r in final], dtype=np.int32)
            new_ids = np.arange(len(final), dtype=np.int32)
            id_remap = np.full(next_label, -1, dtype=np.int32)
            for old, new in zip(old_ids, new_ids):
                id_remap[old] = new

            valid = (label_map >= 0) & (label_map < next_label)
            new_label_map = np.full((h, w), -1, dtype=np.int32)
            new_label_map[valid] = id_remap[label_map[valid]]
            label_map = new_label_map

            for i, r in enumerate(final):
                r.id = i

        # Assign initial heights by luminance
        by_lum = sorted(final, key=lambda r: 0.299 * r.color[0] + 0.587 * r.color[1] + 0.114 * r.color[2])
        for i, r in enumerate(by_lum):
            r.height = int((i / max(1, len(by_lum) - 1)) * 255)

        return DetectionResult(
            label_map=label_map,
            regions=final,
            processor_name=self.name,
        )
