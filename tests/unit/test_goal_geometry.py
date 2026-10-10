"""Tests for field-derived Bonzini B90 goal geometry."""

from dataclasses import replace

import pytest

from app.detection.config import DEFAULT_DETECTION_CONFIG
from app.detection.contracts.table_contracts import FieldGeometry, GoalEnd
from app.detection.goal_geometry import BonziniGoalGeometryEstimator


FIELD = FieldGeometry(
    corners=((150.0, 150.0), (850.0, 150.0), (850.0, 750.0), (150.0, 750.0)),
    bounding_box=(150, 150, 700, 600),
    confidence=0.92,
    detection_method="synthetic",
)


def test_estimator_scales_b90_goal_bounds_from_calibrated_field() -> None:
    """B90 opening width and depth map into the canonical field coordinates."""
    mouths = BonziniGoalGeometryEstimator().estimate(FIELD)

    assert [mouth.end for mouth in mouths] == [GoalEnd.START, GoalEnd.END]
    start, end = mouths
    assert start.opening_bounds == pytest.approx(
        (356.642857, -75.0, 642.357143, 0.0),
        abs=0.001,
    )
    assert end.opening_bounds == pytest.approx(
        (356.642857, 599.0, 642.357143, 674.0),
        abs=0.001,
    )
    assert start.crossing_line_y == 0.0
    assert end.crossing_line_y == 599.0
    assert start.confidence == FIELD.confidence
    assert start.diagnostics == ("bonzini_b90_geometry_estimate",)


def test_estimator_uses_versioned_bonzini_dimensions() -> None:
    """Changing configured physical dimensions changes scaled canonical bounds."""
    config = replace(DEFAULT_DETECTION_CONFIG, bonzini_goal_opening_width_mm=210.0)

    mouths = BonziniGoalGeometryEstimator(config).estimate(FIELD)

    assert mouths[0].opening_bounds[2] - mouths[0].opening_bounds[0] == pytest.approx(300.0)


def test_configured_geometry_is_reproducible_without_frame_pixels() -> None:
    """Goal estimates depend only on accepted field geometry and B90 dimensions."""
    estimator = BonziniGoalGeometryEstimator()

    assert estimator.estimate(FIELD) == estimator.estimate(FIELD)


def test_config_rejects_invalid_bonzini_dimensions() -> None:
    """Physical dimensions must be positive and the goal narrower than the field."""
    with pytest.raises(ValueError, match="must be positive"):
        replace(DEFAULT_DETECTION_CONFIG, bonzini_goal_mouth_depth_mm=0.0)
    with pytest.raises(ValueError, match="narrower"):
        replace(DEFAULT_DETECTION_CONFIG, bonzini_goal_opening_width_mm=700.0)
