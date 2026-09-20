"""Tests for temporal rod candidate consensus."""

import pytest

from app.detection.contracts.rod_contracts import RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.rod_consensus import RodConsensus, RodConsensusError


def _field() -> FieldGeometry:
    """Return a field geometry suitable for consensus contract tests."""
    return FieldGeometry(((0.0, 0.0), (100.0, 0.0), (100.0, 600.0), (0.0, 600.0)), (0, 0, 100, 600), 0.9, "test")


def _candidate(relative_y: float, confidence: float = 0.8) -> RodCandidate:
    """Build a candidate at one normalized field position."""
    return RodCandidate(((0.0, relative_y), (100.0, relative_y)), relative_y, 0.8, 0.2, confidence)


def test_consensus_recovers_rods_missing_from_one_frame() -> None:
    """Stable clusters across frames produce the complete eight-rod layout."""
    positions = (-0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    candidates_by_frame = [
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions if position != 0.35),
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions),
    ]

    rods = RodConsensus().combine(candidates_by_frame, _field())

    assert len(rods) == 8
    assert [rod.index for rod in rods] == list(range(8))
    assert rods[3].field_relative_y == pytest.approx(0.35)


def test_consensus_extends_detected_rod_slope_to_perspective_field_width() -> None:
    """Stable rod output preserves the detected slope while reaching field sides."""
    field = FieldGeometry(
        ((10.0, 10.0), (110.0, 20.0), (140.0, 620.0), (-20.0, 600.0)),
        (-20, 10, 160, 610),
        0.9,
        "test",
    )
    positions = (-0.05, 0.08, 0.22, 0.36, 0.51, 0.66, 0.82, 0.97)

    candidates = tuple(_candidate(position) for position in positions)
    candidates = candidates[:3] + (RodCandidate(((20.0, 220.0), (90.0, 235.0)), 0.36, 0.8, 0.2, 0.9),) + candidates[4:]

    rods = RodConsensus().combine([candidates] * 5, field)

    assert rods[3].line[0] == pytest.approx((-0.455, 215.617), abs=0.01)
    assert rods[3].line[1] == pytest.approx((121.084, 241.661), abs=0.01)


def test_consensus_keeps_outside_goal_rod_on_its_central_detected_fragment() -> None:
    """A goal rod outside the field is not moved onto the field boundary."""
    field = FieldGeometry(
        ((10.0, 10.0), (110.0, 20.0), (140.0, 620.0), (-20.0, 600.0)),
        (-20, 10, 160, 610),
        0.9,
        "test",
    )
    positions = (0.08, 0.22, 0.36, 0.51, 0.66, 0.82, 0.97)
    goal_line = ((15.0, 2.0), (115.0, 17.0))
    central_fragment = ((17.0, 8.0), (117.0, 23.0))
    boundary_fragment = ((20.0, 15.0), (120.0, 30.0))
    frames = [
        (
            RodCandidate(goal_line, -0.05, 0.8, 0.2, 0.7),
            RodCandidate(central_fragment, -0.03, 0.8, 0.2, 0.6),
            RodCandidate(boundary_fragment, -0.01, 0.8, 0.2, 0.95),
            *(_candidate(position) for position in positions),
        )
        for _ in range(5)
    ]

    rods = RodConsensus().combine(frames, field)

    assert rods[0].field_relative_y == pytest.approx(-0.03)
    assert rods[0].line == central_fragment


def test_consensus_rejects_low_coverage_false_positive() -> None:
    """A line seen in only one sampled frame cannot become a stable rod."""
    positions = (-0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    candidates_by_frame = [tuple(_candidate(position) for position in positions) for _ in range(5)]
    candidates_by_frame[0] += (_candidate(0.77),)

    rods = RodConsensus().combine(candidates_by_frame, _field())

    assert len(rods) == 8


def test_consensus_rejects_layout_missing_the_goalkeeper_rod() -> None:
    """An in-field row cannot be silently relabeled as a missing goal rod."""
    positions = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1)

    with pytest.raises(RodConsensusError, match="goal-area envelope"):
        RodConsensus().combine([tuple(_candidate(position) for position in positions)] * 5, _field())


def test_observed_rods_keeps_in_field_rows_when_goalkeeper_validation_fails() -> None:
    """Diagnostics retain stable rows without treating an incomplete layout as calibration."""
    positions = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)

    rods = RodConsensus().observed_rods([tuple(_candidate(position) for position in positions)] * 5, _field())

    assert len(rods) == 7
    assert [rod.field_relative_y for rod in rods] == list(positions)


def test_observed_rods_limits_debug_output_to_eight_strongest_rows() -> None:
    """Diagnostics do not render persistent extras as physical rods."""
    positions = (-0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1)

    rods = RodConsensus().observed_rods([tuple(_candidate(position) for position in positions)] * 5, _field())

    assert len(rods) == 8


def test_consensus_rejects_goal_frame_line_far_above_the_field() -> None:
    """A distant goal-frame edge cannot be used as the goalkeeper rod."""
    positions = (-0.25, -0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8)

    with pytest.raises(RodConsensusError, match="goal-area envelope"):
        RodConsensus().combine([tuple(_candidate(position) for position in positions)] * 5, _field())


def test_consensus_requires_eight_stable_rods() -> None:
    """Incomplete layouts fail with a structured consensus error."""
    with pytest.raises(RodConsensusError, match="Expected 8 stable rods"):
        RodConsensus().combine([tuple(_candidate(position) for position in (0.1, 0.3, 0.5))] * 5, _field())