"""Automatic XRS ROI detection and interactive annotation for Jupyter."""

from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
from matplotlib.path import Path as PolygonPath
from matplotlib.widgets import Button


ROI_MODES = ("free", "rectangle")

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


def detect_candidates(
    image: np.ndarray,
    *,
    roi_mode: str = "free",
    min_area: int = 20,
    max_area: int | None = None,
    threshold_percentile: float = 99.3,
    threshold_sigma: float = 5.0,
    background_sigma: float = 15.0,
    denoise_sigma: float = 1.0,
    morphology_kernel: int = 5,
    approximation_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Detect bright regions as free-shape or rectangular ROI candidates."""

    mode = _normalise_roi_mode(roi_mode)
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
    if not 0.0 <= approximation_ratio <= 0.2:
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


def annotate_candidates(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    roi_mode: str = "free",
    valid_labels: Iterable[str] = VALID_LABELS,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> list[dict[str, Any]]:
    """Assign analyser labels interactively to detected ROI candidates."""

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
    figure = plt.figure(figsize=figsize)
    axis = figure.add_axes((0.05, 0.08, 0.61, 0.86))
    axis.imshow(image, cmap="gray", origin="upper")
    axis.set_xlabel("Pixel x")
    axis.set_ylabel("Pixel y")
    status = axis.set_title(
        f"{mode.capitalize()} ROI: select a candidate, then choose a label"
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
    initial_mode: str = "free",
    valid_labels: Iterable[str] = VALID_LABELS,
    figsize: tuple[float, float] = (8.0, 6.0),
    **detection_kwargs: Any,
) -> dict[str, Any]:
    """Display Jupyter controls for choosing, detecting, and labelling ROIs.

    The returned state dictionary exposes the current ``mode`` and mutable
    ``candidates`` list after the user clicks the detection button.
    """

    mode = _normalise_roi_mode(initial_mode)
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

    mode_dropdown = widgets.Dropdown(
        options=(("Free", "free"), ("Rectangle", "rectangle")),
        value=mode,
        description="ROI mode:",
    )
    run_button = widgets.Button(
        description="Detect and annotate",
        button_style="primary",
        tooltip="Detect candidates with the selected geometry",
    )
    status = widgets.HTML(value="Choose an ROI mode, then run detection.")
    output = widgets.Output()
    state: dict[str, Any] = {
        "mode": mode,
        "candidates": [],
        "mode_dropdown": mode_dropdown,
        "run_button": run_button,
        "output": output,
    }

    def mode_changed(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        state["mode"] = change["new"]
        status.value = (
            f"Selected <b>{change['new']}</b> mode. Run detection to update ROIs."
        )

    def run_detection(_button: Any) -> None:
        selected_mode = mode_dropdown.value
        candidates = detect_candidates(
            image,
            roi_mode=selected_mode,
            **detection_kwargs,
        )
        state["mode"] = selected_mode
        state["candidates"] = candidates
        status.value = (
            f"Detected <b>{len(candidates)}</b> candidate(s) in "
            f"<b>{selected_mode}</b> mode."
        )
        with output:
            clear_output(wait=True)
            if candidates:
                annotate_candidates(
                    image,
                    candidates,
                    roi_mode=selected_mode,
                    valid_labels=valid_labels,
                    figsize=figsize,
                )
            else:
                print("No ROI candidates were detected with the current settings.")

    mode_dropdown.observe(mode_changed, names="value")
    run_button.on_click(run_detection)
    controls = widgets.VBox(
        (widgets.HBox((mode_dropdown, run_button)), status, output)
    )
    state["controls"] = controls
    display(controls)
    return state
