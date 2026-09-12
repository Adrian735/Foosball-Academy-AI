"""Tests for per-frame rod candidate detection."""

import cv2
import numpy as np

from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.rod_detector import RodDetector


def _field() -> FieldGeometry:
    """Return a perspective field geometry for synthetic rod lines."""
    return FieldGeometry(
        corners=((80.0, 40.0), (520.0, 55.0), (550.0, 350.0), (50.0, 335.0)),
        bounding_box=(50, 40, 500, 310),
        confidence=0.9,
        detection_method="synthetic",
    )


def test_detector_returns_scored_horizontal_rod_candidates() -> None:
    """Long bright lines inside the field become field-relative candidates."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.fillConvexPoly(image, np.asarray(_field().corners, dtype=np.int32), (40, 150, 80))
    cv2.line(image, (40, 190), (560, 195), (210, 210, 210), 4)

    candidates = RodDetector().detect(image, _field())

    assert candidates
    assert all(candidate.length_ratio > 0.25 for candidate in candidates)
    assert all(any(item.startswith("angle_score=") for item in candidate.diagnostics) for candidate in candidates)
    assert all(-1.0 <= candidate.field_relative_y <= 2.0 for candidate in candidates)


def test_detector_keeps_rejected_lines_as_json_safe_diagnostics() -> None:
    """Short or non-horizontal Hough lines are inspectable without raw OpenCV values."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.fillConvexPoly(image, np.asarray(_field().corners, dtype=np.int32), (40, 150, 80))
    cv2.line(image, (150, 100), (190, 180), (240, 240, 240), 3)

    detector = RodDetector()
    detector.detect(image, _field())

    assert all(isinstance(candidate.to_dict(), dict) for candidate in detector.rejected_candidates)


def test_detector_allows_shorter_rod_segment_outside_field() -> None:
    """A partially visible goal-area rod can pass the lower outside-field length gate."""
    detector = RodDetector()
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.line(image, (250, 15), (400, 15), (220, 220, 220), 4)
    candidate = detector._score_line(
        image,
        _field(),
        (250, 15, 400, 15),
    )

    assert candidate is not None
    assert candidate.field_relative_y < 0.0
    assert candidate.length_ratio < detector._config.minimum_rod_length_ratio
    assert detector._is_accepted(candidate)