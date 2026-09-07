"""Tests for field candidate consensus and stability checks."""

import pytest

from app.detection.config import DetectionConfig
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.field_consensus import FieldConsensus, FieldConsensusError


def _candidate(offset: float, confidence: float = 0.9) -> FieldGeometry:
    """Create a rectangular candidate with a uniform pixel offset."""
    corners = (
        (100.0 + offset, 50.0 + offset),
        (500.0 + offset, 50.0 + offset),
        (520.0 + offset, 300.0 + offset),
        (80.0 + offset, 300.0 + offset),
    )
    return FieldGeometry(corners, (80, 50, 440, 250), confidence, "synthetic")


def test_consensus_uses_median_corners() -> None:
    """Stable candidates combine into a field near their component-wise median."""
    result = FieldConsensus().combine([_candidate(-2), _candidate(0), _candidate(3), _candidate(1), _candidate(0)])

    assert result.detection_method == "frame_consensus"
    assert result.corners[0] == (100.0, 50.0)
    assert result.bounding_box == (80, 50, 440, 250)


def test_consensus_rejects_too_few_or_unstable_candidates() -> None:
    """Consensus refuses incomplete samples and excessive camera movement."""
    with pytest.raises(FieldConsensusError, match="Too few"):
        FieldConsensus().combine([_candidate(0), _candidate(1)])

    config = DetectionConfig(maximum_field_corner_spread_pixels=5.0)
    with pytest.raises(FieldConsensusError, match="spread"):
        FieldConsensus(config).combine([_candidate(-20), _candidate(20), _candidate(0), _candidate(1), _candidate(2)])