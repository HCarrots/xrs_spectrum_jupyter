import tempfile
import unittest
from pathlib import Path

import numpy as np

from xrs_src.xrs_autoROI import (
    detect_candidates,
    read_rois_txt,
    save_rois_txt,
)


def synthetic_detector() -> tuple[np.ndarray, dict[int, list[tuple[float, float]]]]:
    """Create four noisy detector panels with 15 Gaussian spots per panel."""

    size = 128
    yy, xx = np.indices((size, size))
    rng = np.random.default_rng(20260924)
    image = 1.0 + rng.uniform(0.0, 0.2, (size, size))
    expected: dict[int, list[tuple[float, float]]] = {}
    panels = (
        (0, 62, 0, 62),
        (0, 62, 66, 128),
        (66, 128, 0, 62),
        (66, 128, 66, 128),
    )

    for panel_index, (top, bottom, left, right) in enumerate(panels):
        if panel_index < 2:
            xs = np.linspace(left + 14, right - 14, 3)
            ys = np.linspace(top + 7, bottom - 7, 5)
            sigma_x, sigma_y = 1.8, 1.8
        else:
            xs = np.linspace(left + 7, right - 7, 5)
            ys = np.linspace(top + 14, bottom - 14, 3)
            sigma_x, sigma_y = 1.8, 3.2
        expected[panel_index] = []
        for y_value in ys:
            for x_value in xs:
                expected[panel_index].append((float(x_value), float(y_value)))
                image += 80.0 * np.exp(
                    -0.5
                    * (
                        ((xx - x_value) / sigma_x) ** 2
                        + ((yy - y_value) / sigma_y) ** 2
                    )
                )

    image[62:66, :] = 0.0
    image[:, 62:66] = 0.0
    dead_y = rng.integers(0, size, 250)
    dead_x = rng.integers(0, size, 250)
    image[dead_y, dead_x] = 0.0
    return image.astype(np.float32), expected


class AdaptiveDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image, cls.expected = synthetic_detector()

    def test_adaptive_finds_full_regular_layout(self) -> None:
        candidates = detect_candidates(
            self.image,
            roi_mode="free",
            detection_method="adaptive",
            candidates_per_panel=15,
            min_confidence=0.0,
            min_area=8,
        )
        self.assertEqual(len(candidates), 60)
        for panel_index in range(4):
            panel_candidates = [
                item for item in candidates if item["panel_index"] == panel_index
            ]
            self.assertEqual(len(panel_candidates), 15)
            detected_centres = np.asarray(
                [item["weighted_center"] for item in panel_candidates]
            )
            for expected_centre in self.expected[panel_index]:
                distances = np.linalg.norm(
                    detected_centres - np.asarray(expected_centre), axis=1
                )
                self.assertLess(float(distances.min()), 7.0)

    def test_rectangle_mode_and_txt_round_trip(self) -> None:
        candidates = detect_candidates(
            self.image,
            roi_mode="rectangle",
            min_confidence=0.0,
            min_area=8,
        )
        self.assertEqual(len(candidates), 60)
        self.assertTrue(all("polygon" not in item for item in candidates))
        with tempfile.TemporaryDirectory() as directory:
            output_path = save_rois_txt(
                Path(directory) / "ignored_name.txt",
                candidates,
                roi_mode="rectangle",
                scan_number="15035",
            )
            loaded = read_rois_txt(output_path)
        self.assertEqual(loaded["roi_mode"], "rectangle")
        self.assertEqual(
            [item["bbox"] for item in loaded["candidates"]],
            [item["bbox"] for item in candidates],
        )

    def test_threshold_method_remains_available(self) -> None:
        candidates = detect_candidates(
            self.image,
            roi_mode="rectangle",
            detection_method="threshold",
            min_area=1,
            threshold_percentile=95.0,
            threshold_sigma=1.0,
            morphology_kernel=1,
        )
        self.assertGreater(len(candidates), 0)

    def test_adaptive_denoise_modes_are_available(self) -> None:
        for denoise_method in ("bilateral", "gaussian", "none"):
            with self.subTest(denoise_method=denoise_method):
                candidates = detect_candidates(
                    self.image,
                    roi_mode="rectangle",
                    denoise_method=denoise_method,
                    min_confidence=0.0,
                    min_area=8,
                )
                self.assertEqual(len(candidates), 60)

    def test_invalid_adaptive_controls_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect_candidates(self.image, detection_method="unknown")
        with self.assertRaises(ValueError):
            detect_candidates(self.image, min_confidence=1.1)
        with self.assertRaises(ValueError):
            detect_candidates(self.image, candidates_per_panel=0)
        with self.assertRaises(ValueError):
            detect_candidates(self.image, denoise_method="unknown")
        with self.assertRaises(ValueError):
            detect_candidates(self.image, denoise_strength=0.0)


if __name__ == "__main__":
    unittest.main()
