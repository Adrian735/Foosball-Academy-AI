"""Synthetic tests for stateless ball candidate detection."""

import cv2
import numpy as np
import pytest

from app.ball_tracking.config import BallTrackingConfig
from app.ball_tracking.detector import BallDetector
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.goal_geometry import BonziniGoalGeometryEstimator


FIELD = FieldGeometry(
    corners=((50.0, 50.0), (550.0, 50.0), (550.0, 350.0), (50.0, 350.0)),
    bounding_box=(50, 50, 500, 300),
    confidence=0.9,
    detection_method="synthetic",
)


def _image() -> np.ndarray:
    """Return a dark synthetic frame with a calibrated field region."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (550, 350), (40, 150, 80), -1)
    return image


def test_detector_accepts_a_yellow_ball_inside_the_field() -> None:
    """A circular yellow blob inside the field becomes an accepted candidate."""
    image = _image()
    cv2.circle(image, (300, 200), 12, (0, 220, 220), -1)

    result = BallDetector().detect(image, FIELD)

    assert len(result.accepted) == 1
    candidate = result.accepted[0]
    assert candidate.center == (300.5, 200.5)
    assert candidate.canonical_center == pytest.approx((500.5, 300.5), abs=0.01)
    assert candidate.confidence >= 0.45
    assert result.rejected == ()


def test_detector_rejects_wrong_hue() -> None:
    """A non-yellow object does not create a false ball candidate."""
    image = _image()
    cv2.circle(image, (300, 200), 12, (220, 0, 0), -1)

    result = BallDetector().detect(image, FIELD)

    assert result.accepted == ()


def test_detector_rejects_non_circular_noise() -> None:
    """A long yellow rectangle is retained as rejected diagnostics."""
    image = _image()
    cv2.rectangle(image, (260, 190), (340, 210), (0, 220, 220), -1)

    result = BallDetector().detect(image, FIELD)

    assert result.accepted == ()
    assert any(
        reason in candidate.diagnostics
        for candidate in result.rejected
        for reason in ("insufficient_circularity", "invalid_aspect_ratio")
    )


def test_detector_rejects_out_of_field_blob() -> None:
    """A yellow blob outside the calibrated polygon is masked out safely."""
    image = _image()
    cv2.circle(image, (30, 30), 12, (0, 220, 220), -1)

    result = BallDetector().detect(image, FIELD)

    assert result.accepted == ()
    assert result.rejected == ()


def test_detector_rejects_field_edge_blob() -> None:
    """A yellow blob near the canonical field edge is rejected explicitly."""
    image = _image()
    cv2.circle(image, (65, 200), 12, (0, 220, 220), -1)

    result = BallDetector(BallTrackingConfig(field_edge_margin_canonical=60.0)).detect(image, FIELD)

    assert result.accepted == ()
    assert any("near_field_edge" in candidate.diagnostics for candidate in result.rejected)


def test_detector_tracks_ball_inside_calibrated_goal_aperture() -> None:
    """Only a B90 field-derived goal aperture extends the ball mask beyond the field."""
    image = _image()
    cv2.rectangle(image, (250, 35), (350, 75), (5, 5, 5), -1)
    cv2.circle(image, (300, 42), 8, (0, 220, 220), -1)
    mouths = BonziniGoalGeometryEstimator().estimate(FIELD)

    result = BallDetector().detect(image, FIELD, mouths)

    assert len(result.accepted) == 1
    assert result.accepted[0].canonical_center is not None
    assert result.accepted[0].canonical_center[1] < 0


def test_detector_does_not_expand_tracking_to_uncalibrated_field_exits() -> None:
    """An arbitrary exterior ball remains outside the field-only detection mask."""
    image = _image()
    cv2.circle(image, (300, 30), 8, (0, 220, 220), -1)

    result = BallDetector().detect(image, FIELD)

    assert result.accepted == ()
