"""Tests for JSON-safe table-calibration contracts."""

import json

from app.detection.contracts.contracts import FieldGeometry, FrameQuality, Rod, TableCalibration, VideoMetadata
from app.detection.contracts.rod_contracts import Rod as DirectRod
from app.detection.contracts.table_contracts import (
    FieldGeometry as DirectFieldGeometry,
    GoalEnd,
    GoalMouth,
)
from app.detection.contracts.video_contracts import VideoMetadata as DirectVideoMetadata


def test_contracts_are_available_from_responsibility_specific_modules() -> None:
    """Each contract family can be imported without the compatibility facade."""
    assert DirectVideoMetadata is VideoMetadata
    assert DirectRod is Rod
    assert DirectFieldGeometry is FieldGeometry


def test_table_calibration_serializes_without_image_or_numpy_values() -> None:
    """A complete calibration report is safe to store as JSON."""
    metadata = VideoMetadata(
        path="tests/fixtures/table_detection/clean.mp4",
        frames_per_second=30.0,
        frame_count=150,
        width=1280,
        height=720,
        duration_seconds=5.0,
    )
    quality = FrameQuality(0, 0.0, 115.0, 140.0, True)
    field = FieldGeometry(
        corners=((100.0, 80.0), (1180.0, 80.0), (1180.0, 650.0), (100.0, 650.0)),
        bounding_box=(100, 80, 1080, 570),
        confidence=0.95,
        detection_method="contour",
    )
    rod = Rod(
        index=0,
        line=((90.0, 130.0), (1190.0, 130.0)),
        field_relative_y=0.09,
        confidence=0.91,
        player_colour_evidence=0.75,
    )
    calibration = TableCalibration(
        metadata=metadata,
        sampled_frame_quality=(quality,),
        field=field,
        rods=(rod,),
        confidence=0.93,
        detector_config_version="1",
    )

    report = calibration.to_dict()

    assert json.loads(json.dumps(report)) == report
    assert report["field"]["corners"][0] == [100.0, 80.0]
    assert report["rods"][0]["index"] == 0


def test_sampled_frame_quality_exposes_a_rejection_reason() -> None:
    """Quality diagnostics retain the reason a frame was not used."""
    quality = FrameQuality(7, 1.2, 3.0, 10.0, False, "blurred")

    assert quality.to_dict()["rejection_reason"] == "blurred"


def test_table_calibration_serializes_detected_goal_mouths_and_diagnostics() -> None:
    """Goal apertures persist independently from field and rod calibration."""
    metadata = VideoMetadata("video.mp4", 30.0, 150, 1280, 720, 5.0)
    field = FieldGeometry(
        ((0.0, 0.0), (1000.0, 0.0), (1000.0, 600.0), (0.0, 600.0)),
        (0, 0, 1000, 600),
        0.9,
        "synthetic",
    )
    mouth = GoalMouth(GoalEnd.START, (390.0, -20.0, 610.0, 5.0), 0.0, 0.88)
    calibration = TableCalibration(
        metadata, (), field, (), 0.9, goal_mouths=(mouth,), goal_warnings=("end_goal_occluded",)
    )

    report = calibration.to_dict()

    assert json.loads(json.dumps(report)) == report
    assert report["goal_mouths"] == [
        {
            "end": "start",
            "opening_bounds": [390.0, -20.0, 610.0, 5.0],
            "crossing_line_y": 0.0,
            "confidence": 0.88,
            "diagnostics": [],
        }
    ]
    assert report["goal_warnings"] == ["end_goal_occluded"]


def test_goal_mouth_rejects_invalid_bounds_and_crossing_line() -> None:
    """Goal geometry requires ordered finite bounds and a valid confidence."""
    import pytest

    with pytest.raises(ValueError, match="opening_bounds"):
        GoalMouth(GoalEnd.START, (610.0, -20.0, 390.0, 5.0), 0.0, 0.8)
    with pytest.raises(ValueError, match="crossing_line_y"):
        GoalMouth(GoalEnd.START, (390.0, -20.0, 610.0, 5.0), float("nan"), 0.8)
    with pytest.raises(ValueError, match="confidence"):
        GoalMouth(GoalEnd.START, (390.0, -20.0, 610.0, 5.0), 0.0, 1.2)
