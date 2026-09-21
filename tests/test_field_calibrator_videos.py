"""Regression checks for supplied field-detection video fixtures."""

import json
from pathlib import Path

import pytest

from app.detection.calibrator import TableCalibrator
from app.detection.contracts.rod_contracts import RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.rod_consensus import RodConsensus, RodConsensusError
from app.detection.rod_detector import RodDetector


VIDEO_DIRECTORY = Path(__file__).parent / "table-detection_tests"


@pytest.mark.parametrize("video_number", (4, 5, 6))
def test_affected_videos_produce_stable_field_calibration(video_number: int) -> None:
    """Distant and diagonal supported fixtures calibrate without review warnings."""
    calibration = TableCalibrator().calibrate_field(
        str(VIDEO_DIRECTORY / f"table-detection_test-{video_number}.mp4")
    )

    assert calibration.field is not None
    assert calibration.confidence >= 0.50
    assert calibration.warnings == ()
    assert calibration.detector_config_version == "3"


def test_calibrate_public_entry_point_returns_field_and_rods() -> None:
    """The public facade coordinates field and strict rod consensus."""
    calibration = TableCalibrator().calibrate(
        str(VIDEO_DIRECTORY / "table-detection_test-1.mp4")
    )

    assert calibration.field is not None
    assert len(calibration.rods) == 8
    assert calibration.warnings == ()


@pytest.mark.parametrize(
    ("video_number", "expects_rods"),
    ((2, False), (4, False), (5, False), (6, True)),
)
def test_calibrate_public_entry_point_routes_ambiguous_layouts_to_review(
    video_number: int,
    expects_rods: bool,
) -> None:
    """The facade never converts an ambiguous rod layout into a success."""
    calibration = TableCalibrator().calibrate(
        str(VIDEO_DIRECTORY / f"table-detection_test-{video_number}.mp4")
    )

    assert bool(calibration.rods) is expects_rods
    if expects_rods:
        assert calibration.warnings == ()
    else:
        assert calibration.field is not None
        assert calibration.warnings
        assert calibration.warnings[0].startswith("Rod consensus requires review:")


@pytest.mark.parametrize("video_number", (1, 6))
def test_annotated_fixtures_produce_eight_rods_within_y_tolerance(video_number: int) -> None:
    """Clean annotated fixtures produce eight consensus rods at expected pixel y positions."""
    calibrator = TableCalibrator()
    video_path = VIDEO_DIRECTORY / f"table-detection_test-{video_number}.mp4"
    expected_path = Path(__file__).parent / "fixtures" / "expected" / f"table-detection_test-{video_number}.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    startup = calibrator._frame_reader.read(str(video_path))
    field_candidates = tuple(
        candidate
        for frame in startup.accepted_frames
        if (candidate := calibrator._field_detector.detect(frame.image)) is not None
    )
    field = calibrator._field_consensus.combine(field_candidates)
    detector = RodDetector()
    candidates_by_frame = tuple(
        detector.detect_frame(frame.image, field).accepted
        for frame in startup.accepted_frames
    )

    rods = RodConsensus().combine(candidates_by_frame, field)
    ordered_rods = sorted(rods, key=lambda rod: rod.field_relative_y)
    actual_y_positions = sorted(
        (rod.line[0][1] + rod.line[1][1]) / 2.0 for rod in ordered_rods[1:-1]
    )
    expected_y_positions = sorted(expected["rod_y_positions"])

    assert len(rods) == expected["expected_rod_count"] == 8
    assert len(ordered_rods[1:-1]) == 6
    assert all(
        abs(actual - target) <= expected["rod_tolerance_px"]
        for actual, target in zip(actual_y_positions, expected_y_positions[1:-1])
    )


def test_annotated_incomplete_fixture_routes_to_review() -> None:
    """A fixture missing a stable rod raises consensus instead of relabeling rows."""
    positions = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)

    geometry = FieldGeometry(((0.0, 0.0), (100.0, 0.0), (100.0, 600.0), (0.0, 600.0)), (0, 0, 100, 600), 0.9, "test")
    candidates = tuple(RodCandidate(((0.0, position), (100.0, position)), position, 0.8, 0.2, 0.8) for position in positions)

    with pytest.raises(RodConsensusError, match="Expected 8 stable rods"):
        RodConsensus().combine([candidates] * 5, geometry)