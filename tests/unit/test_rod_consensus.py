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
    positions = (-0.2, 0.05, 0.2, 0.37, 0.52, 0.69, 0.84, 1.0)
    candidates_by_frame = [
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions if position != 0.37),
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions),
        tuple(_candidate(position) for position in positions),
    ]

    rods = RodConsensus().combine(candidates_by_frame, _field())

    assert len(rods) == 8
    assert [rod.index for rod in rods] == list(range(8))
    assert rods[3].field_relative_y == pytest.approx(0.37)


def test_consensus_rejects_low_coverage_false_positive() -> None:
    """A line seen in only one sampled frame cannot become a stable rod."""
    positions = (-0.2, 0.05, 0.2, 0.37, 0.52, 0.69, 0.84, 1.0)
    candidates_by_frame = [tuple(_candidate(position) for position in positions) for _ in range(5)]
    candidates_by_frame[0] += (_candidate(0.77),)

    rods = RodConsensus().combine(candidates_by_frame, _field())

    assert len(rods) == 8


def test_consensus_requires_eight_stable_rods() -> None:
    """Incomplete layouts fail with a structured consensus error."""
    with pytest.raises(RodConsensusError, match="Expected 8 stable rods"):
        RodConsensus().combine([tuple(_candidate(position) for position in (0.1, 0.3, 0.5))] * 5, _field())