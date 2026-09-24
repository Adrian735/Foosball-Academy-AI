"""Focused tests for standalone ball-tracking contracts and calibration loading."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from app.ball_tracking.calibration import (
    CalibrationLoadError,
    require_trackable_calibration,
    table_calibration_from_dict,
)
from app.ball_tracking.config import BallTrackingConfig, DEFAULT_BALL_TRACKING_CONFIG
from app.ball_tracking.contracts import (
    BallCandidate,
    BallObservation,
    BallTrack,
    ObservationState,
)


def _calibration_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata": {
            "path": "clip.mp4",
            "frames_per_second": 30.0,
            "frame_count": 180,
            "width": 1920,
            "height": 1080,
            "duration_seconds": 6.0,
        },
        "sampled_frame_quality": [],
        "field": {
            "corners": [[100, 100], [1800, 120], [1780, 950], [120, 940]],
            "bounding_box": [100, 100, 1700, 850],
            "confidence": 0.8,
            "detection_method": "contour_approximation",
        },
        "rods": [],
        "confidence": 0.8,
        "warnings": [],
        "detector_config_version": "3",
    }
    payload.update(overrides)
    return payload


def test_ball_contracts_are_json_safe() -> None:
    """Nested ball contracts serialize without OpenCV or NumPy values."""
    candidate = BallCandidate(
        center=(20.0, 30.0),
        canonical_center=(100.0, 200.0),
        bounding_box=(15.0, 25.0, 10.0, 10.0),
        area=75.0,
        circularity=0.9,
        aspect_ratio=1.0,
        confidence=0.85,
    )
    observation = BallObservation(4, 0.2, ObservationState.DETECTED, candidate, 0.85)
    track = BallTrack({"frame_count": 10}, "3", "1", (observation,), 0.1, 0.0, 0.85)

    serialized = json.dumps(track.to_dict())

    assert '"state": "detected"' in serialized
    assert track.to_dict()["observations"][0]["candidate"]["center"] == [20.0, 30.0]


def test_ball_tracking_config_is_frozen_and_overridable() -> None:
    """A test can replace one threshold without mutating global defaults."""
    custom = replace(DEFAULT_BALL_TRACKING_CONFIG, minimum_track_coverage=0.8)

    assert custom.minimum_track_coverage == 0.8
    assert DEFAULT_BALL_TRACKING_CONFIG.minimum_track_coverage == 0.60
    with pytest.raises(FrozenInstanceError):
        custom.minimum_track_coverage = 0.2  # type: ignore[misc]


def test_calibration_loader_accepts_table_calibration_shape() -> None:
    """The loader reconstructs the public table-calibration contract."""
    calibration = table_calibration_from_dict(_calibration_payload())

    assert calibration.field is not None
    assert calibration.field.corners[0] == (100.0, 100.0)
    assert calibration.metadata.width == 1920


def test_calibration_loader_rejects_unknown_keys() -> None:
    """Unknown persisted fields fail instead of being silently ignored."""
    payload = _calibration_payload()
    payload["unexpected"] = True

    with pytest.raises(CalibrationLoadError, match="unknown unexpected"):
        table_calibration_from_dict(payload)


def test_trackable_calibration_rejects_review_warnings() -> None:
    """Calibration warnings prevent ball tracking from guessing a field."""
    calibration = table_calibration_from_dict(_calibration_payload(warnings=["unstable field"]))

    with pytest.raises(CalibrationLoadError, match="requires review"):
        require_trackable_calibration(calibration)


def test_trackable_calibration_rejects_missing_field() -> None:
    """A report without field geometry cannot enter the tracker."""
    calibration = table_calibration_from_dict(_calibration_payload(field=None))

    with pytest.raises(CalibrationLoadError, match="no field geometry"):
        require_trackable_calibration(calibration)
