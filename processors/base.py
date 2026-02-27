"""
Base processor interface.

All region detection / image processing algorithms implement this.
To add a new processor (e.g. AI segmentation):
  1. Create a new file in processors/
  2. Subclass BaseProcessor
  3. Register it in processors/__init__.py
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import cv2


@dataclass
class Region:
    """A detected region in the image."""
    id: int
    color: tuple[int, int, int]       # Average RGB color
    pixel_count: int
    bbox: tuple[int, int, int, int]   # (min_x, min_y, max_x, max_y)
    height: int = 128                  # Assigned grayscale height 0-255
    label: str = ""                    # Optional human-readable label
    metadata: dict = field(default_factory=dict)  # Extra data from processor


@dataclass
class DetectionResult:
    """Output of a processor's detect() method."""
    label_map: np.ndarray              # (H, W) int32 array, pixel → region id
    regions: list[Region]
    processor_name: str
    metadata: dict = field(default_factory=dict)


# ── Common pre/post-processing parameters ──
COMMON_PARAMS = [
    {
        "key": "_pre_blur",
        "label": "Suavizado previo (anti-aliasing)",
        "type": "slider",
        "min": 0, "max": 9, "step": 2,
        "default": 3,
        "hint": "Difumina la imagen antes de detectar para eliminar artefactos de anti-aliasing (0 = desactivado, valores impares)",
        "group": "preprocessing",
    },
    {
        "key": "_label_smooth",
        "label": "Suavizado de regiones",
        "type": "slider",
        "min": 0, "max": 15, "step": 2,
        "default": 5,
        "hint": "Filtro de mediana sobre el mapa de regiones para eliminar líneas finas y artefactos (0 = desactivado, valores impares)",
        "group": "preprocessing",
    },
]


class BaseProcessor(ABC):
    """Interface for region detection algorithms."""

    # Display name shown in the UI
    name: str = "Base"
    description: str = ""

    def get_all_params(self) -> list[dict]:
        """Return processor params + common preprocessing params."""
        return self.get_params() + COMMON_PARAMS

    @abstractmethod
    def get_params(self) -> list[dict]:
        """Return parameter definitions for the UI.

        Each param dict:
            {
                "key": "tolerance",
                "label": "Tolerancia",
                "type": "slider" | "number" | "checkbox" | "select",
                "min": 0, "max": 100, "step": 1,
                "default": 30,
                "hint": "Color distance threshold"
            }
        """
        ...

    @abstractmethod
    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        """Run region detection on an RGBA image.

        Args:
            image: (H, W, 4) uint8 RGBA numpy array
            params: dict of parameter values from the UI

        Returns:
            DetectionResult with label map and region list
        """
        ...

    def run(self, image: np.ndarray, params: dict) -> DetectionResult:
        """Full pipeline: pre-process → detect → post-process → smooth labels."""
        # ── Pre-processing: Gaussian blur to remove anti-aliasing ──
        pre_blur = int(params.get("_pre_blur", 3))
        if pre_blur > 0:
            # Ensure odd kernel size
            if pre_blur % 2 == 0:
                pre_blur += 1
            # Blur only the RGB channels, preserve alpha
            rgb = image[:, :, :3]
            alpha = image[:, :, 3:]
            rgb_blurred = cv2.GaussianBlur(rgb, (pre_blur, pre_blur), 0)
            processed = np.concatenate([rgb_blurred, alpha], axis=2)
        else:
            processed = image

        # ── Detection ──
        result = self.detect(processed, params)
        result = self.post_process(result, image)

        # ── Post-processing: median filter on label map ──
        label_smooth = int(params.get("_label_smooth", 5))
        if label_smooth > 0:
            if label_smooth % 2 == 0:
                label_smooth += 1
            result = self._smooth_labels(result, image, label_smooth)

        return result

    def _smooth_labels(self, result: DetectionResult, image: np.ndarray, kernel_size: int) -> DetectionResult:
        """Apply median filter to the label map to remove thin artifacts.

        This replaces isolated thin-line labels with their surrounding region,
        making the segmentation more homogeneous.
        """
        lm = result.label_map.copy()
        h, w = lm.shape

        # Median filter on label map — treats labels as values and picks
        # the most common label in the neighborhood, effectively removing
        # thin lines and small isolated patches
        # We use cv2.medianBlur which requires uint8, so we remap labels
        unique_labels = np.unique(lm)
        # Handle case with more than 254 labels (unlikely but safe)
        if len(unique_labels) <= 254:
            # Map labels to 0-253 range for uint8 median filter, -1 → 255
            label_to_idx = {}
            idx_to_label = {}
            next_idx = 0
            for lab in unique_labels:
                if lab < 0:
                    label_to_idx[lab] = 255
                    idx_to_label[255] = lab
                else:
                    label_to_idx[lab] = next_idx
                    idx_to_label[next_idx] = lab
                    next_idx += 1

            # Convert to uint8
            lm_u8 = np.zeros((h, w), dtype=np.uint8)
            for lab, idx in label_to_idx.items():
                lm_u8[lm == lab] = idx

            # Apply median filter
            lm_u8 = cv2.medianBlur(lm_u8, kernel_size)

            # Convert back
            new_lm = np.full((h, w), -1, dtype=np.int32)
            for idx, lab in idx_to_label.items():
                new_lm[lm_u8 == idx] = lab

            result.label_map = new_lm

        # Recount pixels per region with smoothed label map
        for r in result.regions:
            r.pixel_count = int(np.sum(result.label_map == r.id))

        # Remove regions that ended up with 0 pixels
        result.regions = [r for r in result.regions if r.pixel_count > 0]

        # Recompute bounding boxes
        for r in result.regions:
            ys, xs = np.where(result.label_map == r.id)
            if len(xs) > 0:
                r.bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        return result

    def post_process(self, result: DetectionResult, image: np.ndarray) -> DetectionResult:
        """Optional post-processing step (merge small regions, smooth, etc.)."""
        return result
