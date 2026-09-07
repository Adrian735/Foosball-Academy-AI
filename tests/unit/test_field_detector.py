"""Tests for single-frame playable-field detection."""

import cv2
import numpy as np

from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.field_detector import FieldDetector


def _synthetic_field() -> np.ndarray:
    """Return a bright synthetic green table on a dark background."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.fillPoly(
        image,
        [np.asarray([[100, 70], [500, 90], [530, 330], [70, 310]], dtype=np.int32)],
        (40, 150, 80),
    )
    return image


def test_detector_returns_ordered_field_geometry() -> None:
    """A clear green quadrilateral produces ordered corners and confidence."""
    result = FieldDetector().detect(_synthetic_field())

    assert isinstance(result, FieldGeometry)
    assert result.confidence >= 0.6
    top_left, top_right, bottom_right, bottom_left = result.corners
    assert top_left[0] < top_right[0]
    assert bottom_left[0] < bottom_right[0]
    assert top_left[1] < bottom_left[1]
    assert top_right[1] < bottom_right[1]


def test_detector_prioritizes_field_geometry_over_image_coverage() -> None:
    """A smaller but rectangular field should retain useful confidence."""
    result = FieldDetector().detect(_synthetic_field())

    assert result is not None
    assert result.confidence >= 0.6


def test_detector_returns_none_when_no_field_colour_is_present() -> None:
    """A dark frame without a field contour is rejected safely."""
    assert FieldDetector().detect(np.zeros((400, 600, 3), dtype=np.uint8)) is None