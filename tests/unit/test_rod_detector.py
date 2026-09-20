"""Tests for per-frame rod candidate detection."""

import cv2
import numpy as np

from app.detection.contracts.rod_contracts import RodCandidate
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


def test_detector_rejects_dark_shadow_band_without_player_evidence() -> None:
    """A broad dark field shadow is not accepted as a physical rod."""
    image = np.full((400, 600, 3), (40, 150, 80), dtype=np.uint8)
    cv2.fillConvexPoly(image, np.asarray(_field().corners, dtype=np.int32), (40, 150, 80))
    cv2.line(image, (40, 190), (560, 195), (5, 25, 15), 16)

    detector = RodDetector()

    candidates = detector.detect(image, _field())

    assert not candidates
    assert any("shadow_band=" in candidate.diagnostics[-1] for candidate in detector.rejected_candidates)


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


def test_detector_merges_short_goal_rod_fragments_before_gating() -> None:
    """Separated goal-rod fragments combine before the outside-field span gate."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.line(image, (130, 15), (230, 15), (220, 220, 220), 4)
    cv2.line(image, (320, 15), (420, 15), (220, 220, 220), 4)

    candidates = RodDetector().detect(image, _field())

    goal_candidates = [candidate for candidate in candidates if candidate.field_relative_y < 0.0]
    assert len(goal_candidates) == 1
    assert goal_candidates[0].length_ratio >= 0.20


def test_detector_does_not_merge_goal_fragments_with_the_field_edge() -> None:
    """Top goal fragments remain separate from a nearby in-field horizontal edge."""
    detector = RodDetector()
    goal = detector._score_line(np.zeros((400, 600, 3), dtype=np.uint8), _field(), (130, 15, 230, 15))
    field_edge = detector._score_line(np.zeros((400, 600, 3), dtype=np.uint8), _field(), (130, 45, 230, 45))

    assert goal is not None
    assert field_edge is not None
    merged = detector._merge_fragments(
        np.zeros((400, 600, 3), dtype=np.uint8),
        _field(),
        [goal, field_edge],
        np.zeros((400, 600, 3), dtype=np.uint8),
        np.zeros((400, 600), dtype=np.uint8),
    )

    assert len(merged) == 2


def test_detector_does_not_deduplicate_goal_and_first_field_rod() -> None:
    """A recovered goal rod is not replaced by the nearby first in-field row."""
    detector = RodDetector()
    goal = RodCandidate(((10.0, 10.0), (110.0, 10.0)), -0.02, 0.8, 0.2, 0.6)
    first_field_rod = RodCandidate(((10.0, 30.0), (110.0, 30.0)), 0.03, 0.8, 0.2, 0.9)

    candidates = detector._deduplicate_candidates([goal, first_field_rod])

    assert candidates == [goal, first_field_rod]