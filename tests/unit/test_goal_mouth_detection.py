"""Synthetic tests for visually detecting the Bonzini goal apertures."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.detection.config import DEFAULT_DETECTION_CONFIG
from app.detection.contracts.table_contracts import FieldGeometry, GoalEnd
from app.detection.goal_mouth_detector import GoalMouthDetector


FIELD = FieldGeometry(
    corners=((150.0, 150.0), (850.0, 150.0), (850.0, 750.0), (150.0, 750.0)),
    bounding_box=(150, 150, 700, 600),
    confidence=0.95,
    detection_method="synthetic",
)


def _table_frame(*, goal_at_start: bool = True, goal_at_end: bool = True) -> np.ndarray:
    """Build a perspective-ready table frame with visually black goal apertures."""
    image = np.full((900, 1000, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (150, 150), (850, 750), (40, 150, 80), -1)
    if goal_at_start:
        cv2.rectangle(image, (400, 118), (600, 165), (5, 5, 5), -1)
    if goal_at_end:
        cv2.rectangle(image, (400, 735), (600, 782), (5, 5, 5), -1)
    return image


def test_detector_visually_finds_both_goal_apertures_in_canonical_field_space() -> None:
    """Dark centered apertures intersecting both field ends produce goal bounds."""
    mouths = GoalMouthDetector().detect_frame(_table_frame(), FIELD)

    assert [mouth.end for mouth in mouths] == [GoalEnd.START, GoalEnd.END]
    start, end = mouths
    assert start.opening_bounds[0] == pytest.approx(357.0, abs=15.0)
    assert start.opening_bounds[2] == pytest.approx(643.0, abs=15.0)
    assert start.opening_bounds[1] < 0.0 < start.opening_bounds[3]
    assert end.opening_bounds[1] < 599.0 < end.opening_bounds[3]
    assert start.crossing_line_y == 0.0
    assert end.crossing_line_y == 599.0
    assert start.confidence >= 0.5
    assert end.confidence >= 0.5


def test_detector_does_not_substitute_missing_goal_apertures() -> None:
    """A dark frame without visible openings yields no inferred geometry."""
    image = np.full((900, 1000, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (150, 150), (850, 750), (40, 150, 80), -1)

    assert GoalMouthDetector().detect_frame(image, FIELD) == ()


def test_detector_rejects_dark_components_away_from_the_goal_center() -> None:
    """A visually dark side feature cannot become a centered goal aperture."""
    image = _table_frame(goal_at_start=False, goal_at_end=False)
    cv2.rectangle(image, (170, 118), (300, 165), (5, 5, 5), -1)

    mouths = GoalMouthDetector().detect_frame(image, FIELD)

    assert mouths == ()


def test_consensus_keeps_a_goal_only_when_repeatedly_visible() -> None:
    """Cross-frame consensus reports one visible end and warns for the occluded end."""
    detector = GoalMouthDetector()
    frame_mouths = (
        detector.detect_frame(_table_frame(), FIELD),
        detector.detect_frame(_table_frame(goal_at_end=False), FIELD),
        detector.detect_frame(_table_frame(), FIELD),
    )

    mouths, warnings = detector.combine(frame_mouths)

    assert [mouth.end for mouth in mouths] == [GoalEnd.START]
    assert "goal_end_aperture_not_confidently_visible" in warnings


def test_detector_configuration_is_used_for_dark_aperture_threshold() -> None:
    """An aperture brighter than the configured dark cue is rejected."""
    config = replace(DEFAULT_DETECTION_CONFIG, goal_aperture_max_value=20)
    image = _table_frame()
    cv2.rectangle(image, (400, 118), (600, 165), (70, 70, 70), -1)

    mouths = GoalMouthDetector(config).detect_frame(image, FIELD)

    assert GoalEnd.START not in [mouth.end for mouth in mouths]
