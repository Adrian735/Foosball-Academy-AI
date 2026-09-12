"""Regression checks for supplied field-detection video fixtures."""

import pytest

from pathlib import Path

from app.detection.calibrator import TableCalibrator


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