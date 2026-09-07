"""Tests for canonical field perspective transforms."""

import pytest

from app.detection.contracts.table_contracts import Corners
from app.detection.geometry import canonical_to_field, field_to_canonical


def test_perspective_transform_round_trip_is_stable() -> None:
    """Mapping a point to the canonical plane and back preserves its location."""
    corners: Corners = ((100.0, 80.0), (900.0, 100.0), (950.0, 580.0), (60.0, 560.0))
    source_point = (450.0, 320.0)

    canonical_point = field_to_canonical(source_point, corners)
    restored_point = canonical_to_field(canonical_point, corners)

    assert 0.0 < canonical_point[0] < 1000.0
    assert 0.0 < canonical_point[1] < 600.0
    assert restored_point[0] == pytest.approx(source_point[0], abs=0.01)
    assert restored_point[1] == pytest.approx(source_point[1], abs=0.01)