import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as PolygonPath
from matplotlib.widgets import Button


def detect_candidates(
    image: np.ndarray,
    *,
    min_area: int = 20,
    max_area: int | None = None,
    threshold_percentile: float = 99.3,
    threshold_sigma: float = 5.0,
    background_sigma: float = 15.0,
    denoise_sigma: float = 1.0,
    morphology_kernel: int = 5,
    approximation_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Detect bright regions and describe each one as a closed free shape."""

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; got shape {image.shape}")
    if min_area < 1:
        raise ValueError("min_area must be at least 1")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be greater than or equal to min_area")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100")
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma cannot be negative")
    if background_sigma <= 0 or denoise_sigma <= 0:
        raise ValueError("Gaussian sigma values must be positive")
    if approximation_ratio < 0 or approximation_ratio > 0.2:
        raise ValueError("approximation_ratio must be between 0 and 0.2")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    denoised = cv2.GaussianBlur(image, (0, 0), denoise_sigma)
    background = cv2.GaussianBlur(denoised, (0, 0), background_sigma)
    signal = denoised - background
    threshold = _robust_threshold(signal, threshold_percentile, threshold_sigma)
    if float(np.max(signal, initial=0.0)) <= 0.0 or threshold <= 0.0:
        return []

    mask = (signal >= threshold).astype(np.uint8)
    kernel_size = max(1, int(morphology_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates: list[dict[str, Any]] = []

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or (max_area is not None and area > max_area):
            continue

        component_mask = component_map == component_id
        contour_input = component_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            contour_input,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        if perimeter <= 0 or contour_area <= 0:
            continue

        polygon_array = _closed_polygon(contour, approximation_ratio)
        polygon = tuple((float(x), float(y)) for x, y in polygon_array)
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        ys, xs = np.nonzero(component_mask)
        points_xy = np.column_stack((xs, ys))
        corrected_values = np.maximum(signal[component_mask], 0.0).astype(
            np.float64
        )

        geometric_center = (float(xs.mean()), float(ys.mean()))
        corrected_sum = float(corrected_values.sum())
        if corrected_sum > 0:
            weighted_center = (
                float(np.dot(xs, corrected_values) / corrected_sum),
                float(np.dot(ys, corrected_values) / corrected_sum),
            )
        else:
            weighted_center = geometric_center

        major, minor, pca_ratio = _pca_features(points_xy)
        circularity = float(4.0 * np.pi * contour_area / perimeter**2)
        candidates.append(
            {
                "candidate_id": "",
                "label": None,
                "polygon": polygon,
                "center": geometric_center,
                "weighted_center": weighted_center,
                "bbox": (x, y, width, height),
                "area": area,
                "contour_area": contour_area,
                "max_intensity": float(image[component_mask].max()),
                "total_intensity": float(image[component_mask].sum(dtype=np.float64)),
                "background_corrected_intensity": corrected_sum,
                "perimeter": perimeter,
                "circularity": circularity,
                "aspect_ratio": max(width, height) / max(min(width, height), 1),
                "pca_major": major,
                "pca_minor": minor,
                "pca_ratio": pca_ratio,
            }
        )

    candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"candidate_{index:03d}"
    return candidates
