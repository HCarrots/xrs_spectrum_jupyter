"""Automatic XRS ROI detection and interactive annotation for Jupyter."""

import csv
import io
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
from matplotlib.path import Path as PolygonPath
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.widgets import Button


ROI_MODES = ("free", "rectangle")
LOG_FLOOR = 1e-6
ROI_TXT_VERSION = 1
DEFAULT_ROI_DIR = Path(__file__).resolve().parent.parent / "ROI"
ROI_TXT_FIELDS = (
    "candidate_id",
    "label",
    "roi_mode",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "center_x",
    "center_y",
    "weighted_center_x",
    "weighted_center_y",
    "area",
    "contour_area",
    "max_intensity",
    "total_intensity",
    "background_corrected_intensity",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "pca_major",
    "pca_minor",
    "pca_ratio",
    "polygon_xy",
)

VALID_LABELS = (
    "VB-A1", "VB-A2", "VB-A3", "VB-B1", "VB-B2", "VB-B3",
    "VB-C1", "VB-C2", "VB-C3", "VB-D1", "VB-D2", "VB-D3",
    "VB-E1", "VB-E2", "VB-E3",
    "HL-A1", "HL-A2", "HL-A3", "HL-B1", "HL-B2", "HL-B3",
    "HL-C1", "HL-C2", "HL-C3", "HL-D1", "HL-D2", "HL-D3",
    "HL-E1", "HL-E2", "HL-E3",
    "VU-A1", "VU-A2", "VU-A3", "VU-B1", "VU-B2", "VU-B3",
    "VU-C1", "VU-C2", "VU-C3", "VU-D1", "VU-D2", "VU-D3",
    "VU-E1", "VU-E2", "VU-E3",
    "VD-A1", "VD-A2", "VD-A3", "VD-B1", "VD-B2", "VD-B3",
    "VD-C1", "VD-C2", "VD-C3", "VD-D1", "VD-D2", "VD-D3",
    "VD-E1", "VD-E2", "VD-E3",
)


def _normalise_roi_mode(roi_mode: str) -> str:
    """Validate and normalise an ROI geometry mode."""

    mode = str(roi_mode).strip().lower()
    if mode not in ROI_MODES:
        choices = ", ".join(ROI_MODES)
        raise ValueError(f"roi_mode must be one of: {choices}")
    return mode


def scan_number_from_source(scan_source: str | Path | int) -> str:
    """Extract the leading scan number from an id or NeXus file path."""

    source = str(scan_source).strip()
    if not source:
        raise ValueError("scan_source cannot be empty")
    candidates = (Path(source).name, Path(source).parent.name, source)
    for candidate in candidates:
        match = re.match(r"^(\d+)(?:_|$)", candidate)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot determine scan number from {scan_source!r}")


def default_roi_path(scan_source: str | Path | int) -> Path:
    """Return ``<scan_number>_ROI.txt`` in the project's existing ROI folder."""

    scan_number = scan_number_from_source(scan_source)
    return DEFAULT_ROI_DIR / f"{scan_number}_ROI.txt"


def _txt_path(output_path: str | Path) -> Path:
    """Return a path with a case-insensitive TXT extension."""

    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")
    return path


def _format_float(value: Any, *, allow_infinite: bool = False) -> str:
    """Format one numeric value for lossless text round-tripping."""

    number = float(value)
    if np.isnan(number) or (np.isinf(number) and not allow_infinite):
        raise ValueError(f"ROI values must be finite; got {value!r}")
    return format(number, ".17g")


def _format_polygon(points: Iterable[Iterable[Any]]) -> str:
    """Encode polygon vertices as semicolon-separated x,y pairs."""

    encoded = []
    for point in points:
        coordinates = tuple(point)
        if len(coordinates) != 2:
            raise ValueError("Each polygon point must contain exactly x and y")
        encoded.append(
            f"{_format_float(coordinates[0])},{_format_float(coordinates[1])}"
        )
    return ";".join(encoded)


def _parse_polygon(value: str) -> tuple[tuple[float, float], ...]:
    """Decode semicolon-separated x,y polygon vertices."""

    if not value:
        return ()
    points = []
    for encoded_point in value.split(";"):
        coordinates = encoded_point.split(",")
        if len(coordinates) != 2:
            raise ValueError(f"Invalid polygon point: {encoded_point!r}")
        point = (float(coordinates[0]), float(coordinates[1]))
        if not np.isfinite(point).all():
            raise ValueError(f"Polygon coordinates must be finite: {encoded_point!r}")
        points.append(point)
    return tuple(points)


def save_rois_txt(
    output_path: str | Path,
    candidates: Iterable[dict[str, Any]],
    *,
    roi_mode: str,
    scan_number: str | int,
) -> Path:
    """Save ROI candidates to a versioned tab-separated text file."""

    mode = _normalise_roi_mode(roi_mode)
    scan_number = scan_number_from_source(scan_number)
    path = _txt_path(output_path)
    path = path.with_name(f"{scan_number}_ROI.txt")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"ROI output directory does not exist: {path.parent}")

    rows = []
    for candidate in candidates:
        required = {
            "candidate_id", "label", "bbox", "center", "weighted_center",
            "area", "contour_area", "max_intensity", "total_intensity",
            "background_corrected_intensity", "perimeter", "circularity",
            "aspect_ratio", "pca_major", "pca_minor", "pca_ratio",
        }
        missing = sorted(required.difference(candidate))
        if missing:
            raise ValueError(f"ROI candidate is missing fields: {', '.join(missing)}")

        bbox = tuple(candidate["bbox"])
        center = tuple(candidate["center"])
        weighted_center = tuple(candidate["weighted_center"])
        if len(bbox) != 4 or len(center) != 2 or len(weighted_center) != 2:
            raise ValueError("ROI bbox and center coordinates have invalid lengths")

        polygon = ""
        if mode == "free":
            if "polygon" not in candidate:
                raise ValueError("Free ROI candidates must contain polygon geometry")
            polygon = _format_polygon(candidate["polygon"])
            if not polygon:
                raise ValueError("Free ROI polygons cannot be empty")

        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "label": "" if candidate["label"] is None else str(candidate["label"]),
                "roi_mode": mode,
                "bbox_x": str(int(bbox[0])),
                "bbox_y": str(int(bbox[1])),
                "bbox_width": str(int(bbox[2])),
                "bbox_height": str(int(bbox[3])),
                "center_x": _format_float(center[0]),
                "center_y": _format_float(center[1]),
                "weighted_center_x": _format_float(weighted_center[0]),
                "weighted_center_y": _format_float(weighted_center[1]),
                "area": str(int(candidate["area"])),
                "contour_area": _format_float(candidate["contour_area"]),
                "max_intensity": _format_float(candidate["max_intensity"]),
                "total_intensity": _format_float(candidate["total_intensity"]),
                "background_corrected_intensity": _format_float(
                    candidate["background_corrected_intensity"]
                ),
                "perimeter": _format_float(candidate["perimeter"]),
                "circularity": _format_float(candidate["circularity"]),
                "aspect_ratio": _format_float(candidate["aspect_ratio"]),
                "pca_major": _format_float(candidate["pca_major"]),
                "pca_minor": _format_float(candidate["pca_minor"]),
                "pca_ratio": _format_float(
                    candidate["pca_ratio"], allow_infinite=True
                ),
                "polygon_xy": polygon,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# XRS_ROI_TXT_VERSION={ROI_TXT_VERSION}\n")
        handle.write(f"# scan_number={scan_number}\n")
        handle.write(f"# roi_mode={mode}\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=ROI_TXT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rois_txt(input_path: str | Path) -> dict[str, Any]:
    """Parse a versioned ROI TXT file written by :func:`save_rois_txt`."""

    path = _txt_path(input_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    data_lines = []
    for line in lines:
        if line.startswith("# "):
            key, separator, value = line[2:].partition("=")
            if not separator:
                raise ValueError(f"Invalid ROI TXT metadata line: {line!r}")
            metadata[key] = value
        elif line.strip():
            data_lines.append(line)

    try:
        version = int(metadata["XRS_ROI_TXT_VERSION"])
        scan_number = scan_number_from_source(metadata["scan_number"])
        mode = _normalise_roi_mode(metadata["roi_mode"])
    except KeyError as exc:
        raise ValueError(f"Missing ROI TXT metadata: {exc.args[0]}") from exc
    if version != ROI_TXT_VERSION:
        raise ValueError(f"Unsupported ROI TXT version: {version}")
    if scan_number_from_source(path.name) != scan_number:
        raise ValueError("ROI TXT filename does not match its scan_number metadata")
    if not data_lines:
        raise ValueError("ROI TXT file is missing its column header")

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)), delimiter="\t")
    if tuple(reader.fieldnames or ()) != ROI_TXT_FIELDS:
        raise ValueError("ROI TXT column header does not match the supported format")

    candidates = []
    for row in reader:
        if row["roi_mode"] != mode:
            raise ValueError("ROI row mode does not match file metadata")
        candidate = {
            "candidate_id": row["candidate_id"],
            "label": row["label"] or None,
            "bbox": (
                int(row["bbox_x"]),
                int(row["bbox_y"]),
                int(row["bbox_width"]),
                int(row["bbox_height"]),
            ),
            "center": (float(row["center_x"]), float(row["center_y"])),
            "weighted_center": (
                float(row["weighted_center_x"]),
                float(row["weighted_center_y"]),
            ),
            "area": int(row["area"]),
            "contour_area": float(row["contour_area"]),
            "max_intensity": float(row["max_intensity"]),
            "total_intensity": float(row["total_intensity"]),
            "background_corrected_intensity": float(
                row["background_corrected_intensity"]
            ),
            "perimeter": float(row["perimeter"]),
            "circularity": float(row["circularity"]),
            "aspect_ratio": float(row["aspect_ratio"]),
            "pca_major": float(row["pca_major"]),
            "pca_minor": float(row["pca_minor"]),
            "pca_ratio": float(row["pca_ratio"]),
        }
        finite_numeric_values = [
            value
            for key, value in candidate.items()
            if key not in {"candidate_id", "label", "bbox", "pca_ratio"}
            for value in (value if isinstance(value, tuple) else (value,))
        ]
        if not np.isfinite(finite_numeric_values).all():
            raise ValueError("ROI TXT contains non-finite numeric values")
        if np.isnan(candidate["pca_ratio"]) or candidate["pca_ratio"] == -np.inf:
            raise ValueError("ROI TXT contains an invalid PCA ratio")
        if mode == "free":
            polygon = _parse_polygon(row["polygon_xy"])
            if not polygon:
                raise ValueError("Free ROI polygons cannot be empty")
            candidate["polygon"] = polygon
        elif row["polygon_xy"]:
            raise ValueError("Rectangle ROI rows cannot contain polygon geometry")
        candidates.append(candidate)

    return {
        "format_version": version,
        "scan_number": scan_number,
        "roi_mode": mode,
        "candidates": candidates,
        "path": path,
    }


def _positive_image(image: np.ndarray) -> np.ndarray:
    """Return finite positive data suitable for logarithmic processing."""

    image = np.asarray(image, dtype=np.float32)
    finite_positive = image[np.isfinite(image) & (image > 0)]
    positive_max = (
        float(finite_positive.max()) if finite_positive.size else LOG_FLOOR
    )
    positive = np.nan_to_num(
        image,
        nan=LOG_FLOOR,
        posinf=positive_max,
        neginf=LOG_FLOOR,
    )
    positive[positive <= 0] = LOG_FLOOR
    return positive


def _prepare_log_data(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return safe positive intensities and their base-10 logarithm."""

    positive = _positive_image(image)
    return positive, np.log10(positive)


def _plot_log_image(
    image: np.ndarray,
    *,
    figsize: tuple[float, float],
    detector_label: str,
) -> tuple[Any, Any]:
    """Plot detector data with the logarithmic style used by ``xrs_IO``."""

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; got shape {image.shape}")
    figure = plt.figure(figsize=figsize)
    axis = figure.add_axes((0.05, 0.08, 0.61, 0.86))
    image_artist = axis.imshow(
        _positive_image(image),
        cmap="viridis",
        origin="upper",
        norm=LogNorm(),
    )
    colorbar = figure.colorbar(image_artist, ax=axis)
    formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
    formatter._useMathText = False
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label(f"{detector_label} sum value (log scale)")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    return figure, axis


def _robust_threshold(
    signal: np.ndarray,
    threshold_percentile: float,
    threshold_sigma: float,
) -> float:
    """Combine a high percentile and a median-absolute-deviation threshold."""

    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    percentile_level = float(np.percentile(finite, threshold_percentile))
    noise_level = median + threshold_sigma * robust_sigma
    return max(0.0, percentile_level, noise_level)


def _pca_features(points_xy: np.ndarray) -> tuple[float, float, float]:
    """Return PCA major spread, minor spread, and their ratio."""

    if len(points_xy) < 2:
        return 0.0, 0.0, 1.0
    centered = points_xy.astype(np.float64) - points_xy.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(
        np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0)
    )[::-1]
    major = float(np.sqrt(eigenvalues[0]))
    minor = float(np.sqrt(eigenvalues[1])) if len(eigenvalues) > 1 else 0.0
    ratio = major / minor if minor > 0 else float("inf")
    return major, minor, ratio


def _closed_polygon(contour: np.ndarray, approximation_ratio: float) -> np.ndarray:
    """Optionally simplify a contour and explicitly close its free shape."""

    perimeter = float(cv2.arcLength(contour, True))
    epsilon = approximation_ratio * perimeter
    approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approximated) < 3:
        x, y, width, height = cv2.boundingRect(contour)
        approximated = np.array(
            [
                (x, y),
                (x + width - 1, y),
                (x + width - 1, y + height - 1),
                (x, y + height - 1),
            ],
            dtype=np.int32,
        )
    if not np.array_equal(approximated[0], approximated[-1]):
        approximated = np.vstack((approximated, approximated[0]))
    return approximated.astype(np.float64)


def _validate_detection_parameters(
    *,
    min_area: int,
    max_area: int | None,
    threshold_percentile: float,
    threshold_sigma: float,
    background_sigma: float,
    denoise_sigma: float,
    morphology_kernel: int,
    approximation_ratio: float,
) -> None:
    """Validate public ROI candidate detection parameters."""

    if not isinstance(min_area, (int, np.integer)) or min_area < 1:
        raise ValueError("min_area must be an integer of at least 1")
    if max_area is not None:
        if not isinstance(max_area, (int, np.integer)):
            raise TypeError("max_area must be an integer or None")
        if max_area < min_area:
            raise ValueError("max_area must be greater than or equal to min_area")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100")
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma cannot be negative")
    if background_sigma <= 0 or denoise_sigma <= 0:
        raise ValueError("Gaussian sigma values must be positive")
    if not isinstance(morphology_kernel, (int, np.integer)):
        raise TypeError("morphology_kernel must be an integer")
    if morphology_kernel < 1:
        raise ValueError("morphology_kernel must be at least 1")
    if not 0.0 <= approximation_ratio <= 0.2:
        raise ValueError("approximation_ratio must be between 0 and 0.2")


def detect_candidates(
    image: np.ndarray,
    *,
    roi_mode: str = "free",
    min_area: int = 20,
    max_area: int | None = None,
    threshold_percentile: float = 98.5,
    threshold_sigma: float = 3.0,
    background_sigma: float = 15.0,
    denoise_sigma: float = 1.0,
    morphology_kernel: int = 3,
    approximation_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Detect bright regions as free-shape or rectangular ROI candidates.

    Detection uses one base-10 log transform after non-positive and invalid
    values are replaced safely. Reported intensity summaries remain linear.
    """

    mode = _normalise_roi_mode(roi_mode)
    source_image = np.asarray(image, dtype=np.float32)
    if source_image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; got shape {source_image.shape}")
    _validate_detection_parameters(
        min_area=min_area,
        max_area=max_area,
        threshold_percentile=threshold_percentile,
        threshold_sigma=threshold_sigma,
        background_sigma=background_sigma,
        denoise_sigma=denoise_sigma,
        morphology_kernel=morphology_kernel,
        approximation_ratio=approximation_ratio,
    )

    intensity_image, detection_image = _prepare_log_data(source_image)
    denoised = cv2.GaussianBlur(detection_image, (0, 0), denoise_sigma)
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
        contours, _ = cv2.findContours(
            component_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE if mode == "free" else cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        if mode == "free" and (perimeter <= 0 or contour_area <= 0):
            continue

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
        circularity = (
            float(4.0 * np.pi * contour_area / perimeter**2)
            if perimeter > 0
            else 0.0
        )
        candidate = {
            "candidate_id": "",
            "label": None,
            "center": geometric_center,
            "weighted_center": weighted_center,
            "bbox": (x, y, width, height),
            "area": area,
            "contour_area": contour_area,
            "max_intensity": float(intensity_image[component_mask].max()),
            "total_intensity": float(
                intensity_image[component_mask].sum(dtype=np.float64)
            ),
            "background_corrected_intensity": corrected_sum,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": max(width, height) / max(min(width, height), 1),
            "pca_major": major,
            "pca_minor": minor,
            "pca_ratio": pca_ratio,
        }
        if mode == "free":
            polygon_array = _closed_polygon(contour, approximation_ratio)
            candidate["polygon"] = tuple(
                (float(x_value), float(y_value))
                for x_value, y_value in polygon_array
            )
        candidates.append(candidate)

    candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"candidate_{index:03d}"
    return candidates


def _candidate_at(
    candidates: Iterable[dict[str, Any]],
    x: float,
    y: float,
    roi_mode: str,
) -> dict[str, Any] | None:
    """Return the smallest candidate geometry containing a point."""

    mode = _normalise_roi_mode(roi_mode)
    matches = []
    for candidate in candidates:
        if mode == "free":
            path = PolygonPath(np.asarray(candidate["polygon"]), closed=True)
            if path.contains_point((x, y), radius=1.0):
                matches.append(candidate)
        else:
            left, top, width, height = candidate["bbox"]
            if left <= x <= left + width and top <= y <= top + height:
                matches.append(candidate)
    return min(matches, key=lambda item: item["area"], default=None)


def load_rois_txt(
    image: np.ndarray,
    *,
    scan_source: str | Path | int | None = None,
    input_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    detector_label: str = "D_LAMBDA",
) -> None:
    """Read saved ROIs and display them over the corresponding detector image."""

    if input_path is None:
        if scan_source is None:
            raise ValueError("Provide scan_source or input_path to load ROI TXT data")
        input_path = default_roi_path(scan_source)
    document = read_rois_txt(input_path)
    mode = document["roi_mode"]
    candidates = document["candidates"]
    figure, axis = _plot_log_image(
        image,
        figsize=figsize,
        detector_label=detector_label,
    )

    for candidate in candidates:
        if mode == "free":
            patch = PolygonPatch(
                np.asarray(candidate["polygon"]),
                closed=True,
                fill=False,
                edgecolor="lime",
                linewidth=1.8,
            )
            text_x, text_y = candidate["center"]
            horizontal_alignment = "center"
        else:
            left, top, width, height = candidate["bbox"]
            patch = Rectangle(
                (left, top),
                width,
                height,
                fill=False,
                edgecolor="lime",
                linewidth=1.8,
            )
            text_x, text_y = left, top
            horizontal_alignment = "left"
        axis.add_patch(patch)
        axis.text(
            text_x,
            text_y,
            candidate["label"] or candidate["candidate_id"],
            color="yellow",
            fontsize=7,
            ha=horizontal_alignment,
            va="bottom",
        )

    axis.set_title(
        f"Saved {mode} ROIs for scan {document['scan_number']} "
        f"({len(candidates)} ROI(s))"
    )
    plt.show()
    return None


def annotate_candidates(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    roi_mode: str = "free",
    valid_labels: Iterable[str] = VALID_LABELS,
    figsize: tuple[float, float] = (8.0, 6.0),
    detector_label: str = "D_LAMBDA",
) -> list[dict[str, Any]]:
    """Assign labels on a LogNorm image styled like ``xrs_IO.plot_det_image``."""

    mode = _normalise_roi_mode(roi_mode)
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; got shape {image.shape}")
    if not candidates:
        return candidates
    if mode == "free" and any("polygon" not in item for item in candidates):
        raise ValueError("Free ROI candidates must contain polygon geometry")

    valid_labels = tuple(valid_labels)
    selected: dict[str, dict[str, Any] | None] = {"candidate": None}
    figure, axis = _plot_log_image(
        image,
        figsize=figsize,
        detector_label=detector_label,
    )
    status = axis.set_title(
        f"{mode.capitalize()} ROI on summed {detector_label}: "
        "select a candidate, then choose a label"
    )

    patches: dict[str, Any] = {}
    texts: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if mode == "free":
            patch = PolygonPatch(
                np.asarray(candidate["polygon"]),
                closed=True,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
            text_x, text_y = candidate["center"]
            horizontal_alignment = "center"
        else:
            left, top, width, height = candidate["bbox"]
            patch = Rectangle(
                (left, top),
                width,
                height,
                fill=False,
                edgecolor="cyan",
                linewidth=1.3,
            )
            text_x, text_y = left, top
            horizontal_alignment = "left"
        axis.add_patch(patch)
        text = axis.text(
            text_x,
            text_y,
            candidate_id,
            color="yellow",
            fontsize=7,
            ha=horizontal_alignment,
            va="bottom",
        )
        patches[candidate_id] = patch
        texts[candidate_id] = text

    label_groups: dict[str, list[str]] = {}
    for label in valid_labels:
        group = label.split("-", 1)[0]
        label_groups.setdefault(group, []).append(label)

    panel_left = 0.69
    panel_width = 0.29
    group_width = panel_width / max(len(label_groups), 1)
    button_width = group_width - 0.008
    label_buttons: dict[str, Button] = {}
    for group_index, (group, labels) in enumerate(label_groups.items()):
        left = panel_left + group_index * group_width
        figure.text(
            left + button_width / 2,
            0.925,
            group,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        for row_index, label in enumerate(labels):
            button_axis = figure.add_axes(
                (left, 0.88 - row_index * 0.052, button_width, 0.038)
            )
            label_buttons[label] = Button(
                button_axis,
                label,
                color="0.88",
                hovercolor="0.96",
            )

    clear_button = Button(
        figure.add_axes((panel_left, 0.045, 0.13, 0.052)), "Clear selected"
    )
    finish_button = Button(
        figure.add_axes((panel_left + 0.15, 0.045, 0.13, 0.052)), "Finish"
    )

    def refresh_styles() -> None:
        current = selected["candidate"]
        assigned = {
            item["label"] for item in candidates if item["label"] is not None
        }
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            patch = patches[candidate_id]
            if candidate is current:
                patch.set_edgecolor("red")
                patch.set_linewidth(2.5)
            elif candidate["label"]:
                patch.set_edgecolor("lime")
                patch.set_linewidth(1.8)
            else:
                patch.set_edgecolor("cyan")
                patch.set_linewidth(1.5 if mode == "free" else 1.3)
            texts[candidate_id].set_text(candidate["label"] or candidate_id)

        for label, button in label_buttons.items():
            if label in assigned:
                button.color = "lightgreen"
                button.hovercolor = "palegreen"
            else:
                button.color = "0.88"
                button.hovercolor = "0.96"
            button.ax.set_facecolor(button.color)
        figure.canvas.draw_idle()

    def select_candidate(event: Any) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        candidate = _candidate_at(candidates, event.xdata, event.ydata, mode)
        if candidate is None:
            return
        selected["candidate"] = candidate
        status.set_text(
            f"Selected {candidate['label'] or candidate['candidate_id']}"
        )
        refresh_styles()

    def assign_label(label: str) -> None:
        candidate = selected["candidate"]
        if candidate is None:
            status.set_text("Select an ROI before choosing a label")
            figure.canvas.draw_idle()
            return
        duplicate = next(
            (
                item
                for item in candidates
                if item is not candidate and item["label"] == label
            ),
            None,
        )
        if duplicate is not None:
            status.set_text(f"{label} is already assigned")
            figure.canvas.draw_idle()
            return
        candidate["label"] = label
        status.set_text(f"Assigned {label}")
        refresh_styles()

    def clear_selected(_event: Any) -> None:
        candidate = selected["candidate"]
        if candidate is None:
            status.set_text("Select an ROI before clearing its label")
            figure.canvas.draw_idle()
            return
        candidate["label"] = None
        status.set_text("Selected ROI label cleared")
        refresh_styles()

    def finish(_event: Any) -> None:
        plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", select_candidate)
    for label, button in label_buttons.items():
        button.on_clicked(
            lambda _event, selected_label=label: assign_label(selected_label)
        )
    clear_button.on_clicked(clear_selected)
    finish_button.on_clicked(finish)
    refresh_styles()
    plt.show()
    return candidates


def interactive_roi(
    image: np.ndarray,
    *,
    scan_source: str | Path | int,
    roi_mode: str = "free",
    min_area: int = 20,
    max_area: int | None = None,
    threshold_percentile: float = 98.5,
    threshold_sigma: float = 3.0,
    background_sigma: float = 15.0,
    denoise_sigma: float = 1.0,
    morphology_kernel: int = 3,
    approximation_ratio: float = 0.0,
    valid_labels: Iterable[str] = VALID_LABELS,
    figsize: tuple[float, float] = (8.0, 6.0),
    detector_label: str = "D_LAMBDA",
) -> dict[str, Any]:
    """Display Jupyter controls for detecting, labelling, and saving ROIs.

    The returned state dictionary exposes the current ``mode`` and mutable
    ``candidates`` list after the user clicks the detection button.
    """

    mode = _normalise_roi_mode(roi_mode)
    _validate_detection_parameters(
        min_area=min_area,
        max_area=max_area,
        threshold_percentile=threshold_percentile,
        threshold_sigma=threshold_sigma,
        background_sigma=background_sigma,
        denoise_sigma=denoise_sigma,
        morphology_kernel=morphology_kernel,
        approximation_ratio=approximation_ratio,
    )
    scan_number = scan_number_from_source(scan_source)
    output_path = default_roi_path(scan_number)
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image; got shape {image.shape}")

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError(
            "interactive_roi requires ipywidgets and IPython in the notebook kernel"
        ) from exc

    mode_label = widgets.HTML(value=f"ROI mode: <b>{mode}</b>")
    threshold_percentile_input = widgets.BoundedFloatText(
        value=threshold_percentile,
        min=0.1,
        max=99.9,
        step=0.1,
        description="Percentile:",
    )
    threshold_sigma_input = widgets.BoundedFloatText(
        value=threshold_sigma,
        min=0.0,
        max=100.0,
        step=0.5,
        description="Sigma factor:",
    )
    min_area_input = widgets.BoundedIntText(
        value=min_area,
        min=1,
        max=1_000_000,
        description="Min area:",
    )
    max_area_input = widgets.BoundedIntText(
        value=0 if max_area is None else max_area,
        min=0,
        max=1_000_000,
        description="Max area:",
        tooltip="Use 0 for no maximum area",
    )
    approximation_input = widgets.BoundedFloatText(
        value=approximation_ratio,
        min=0.0,
        max=0.2,
        step=0.005,
        description="Simplify:",
        disabled=mode != "free",
    )
    run_button = widgets.Button(
        description="Detect and annotate",
        button_style="primary",
        tooltip="Detect candidates with the selected geometry",
    )
    output_name_field = widgets.Text(
        value=output_path.name,
        description="TXT file:",
        tooltip=f"Saved inside {DEFAULT_ROI_DIR}",
        disabled=True,
    )
    save_button = widgets.Button(
        description="Save TXT",
        button_style="success",
        tooltip="Save the current ROI candidates and labels",
    )
    status = widgets.HTML(value="Adjust detection settings, then run detection.")
    output = widgets.Output()
    state: dict[str, Any] = {
        "mode": mode,
        "scan_number": scan_number,
        "candidates": [],
        "run_button": run_button,
        "threshold_percentile_input": threshold_percentile_input,
        "threshold_sigma_input": threshold_sigma_input,
        "min_area_input": min_area_input,
        "max_area_input": max_area_input,
        "approximation_input": approximation_input,
        "output_name_field": output_name_field,
        "save_button": save_button,
        "output": output,
        "output_path": output_path,
    }

    def run_detection(_button: Any) -> None:
        selected_max_area = max_area_input.value or None
        try:
            candidates = detect_candidates(
                image,
                roi_mode=mode,
                min_area=min_area_input.value,
                max_area=selected_max_area,
                threshold_percentile=threshold_percentile_input.value,
                threshold_sigma=threshold_sigma_input.value,
                background_sigma=background_sigma,
                denoise_sigma=denoise_sigma,
                morphology_kernel=morphology_kernel,
                approximation_ratio=approximation_input.value,
            )
        except (TypeError, ValueError) as exc:
            status.value = f"ROI detection failed: {exc}"
            return
        state["candidates"] = candidates
        status.value = (
            f"Detected <b>{len(candidates)}</b> candidate(s) in "
            f"<b>{mode}</b> mode."
        )
        with output:
            clear_output(wait=True)
            if candidates:
                annotate_candidates(
                    image,
                    candidates,
                    roi_mode=mode,
                    valid_labels=valid_labels,
                    figsize=figsize,
                    detector_label=detector_label,
                )
            else:
                print("No ROI candidates were detected with the current settings.")

    def save_detection(_button: Any) -> None:
        try:
            saved_path = save_rois_txt(
                output_path,
                state["candidates"],
                roi_mode=state["mode"],
                scan_number=scan_number,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            status.value = f"TXT save failed: {exc}"
            return
        state["output_path"] = saved_path
        status.value = (
            f"Saved <b>{len(state['candidates'])}</b> ROI(s) to "
            f"<code>{saved_path}</code>."
        )

    run_button.on_click(run_detection)
    save_button.on_click(save_detection)
    controls = widgets.VBox(
        (
            widgets.HBox((mode_label, run_button)),
            widgets.HBox((threshold_percentile_input, threshold_sigma_input)),
            widgets.HBox((min_area_input, max_area_input, approximation_input)),
            widgets.HBox((output_name_field, save_button)),
            status,
            output,
        )
    )
    state["controls"] = controls
    display(controls)
    return state
