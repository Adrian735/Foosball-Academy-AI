"""Regression coverage for the public table-calibration facade."""

from pathlib import Path

import pytest

from app.detection.calibrator import TableCalibrator
from app.detection.startup_frames import VideoValidationError


VIDEO_DIRECTORY = Path(__file__).parents[1] / "table-detection_tests"


@pytest.mark.parametrize(
    ("video_number", "expects_supported_layout"),
    ((1, True), (2, False), (3, None), (4, False), (5, False), (6, True)),
)
def test_calibrate_regresses_every_supplied_fixture(
    video_number: int,
    expects_supported_layout: bool | None,
) -> None:
    """Run every fixture through the public facade and preserve review routing."""
    video_path = str(VIDEO_DIRECTORY / f"table-detection_test-{video_number}.mp4")
    if expects_supported_layout is None:
        with pytest.raises(VideoValidationError, match="resolution"):
            TableCalibrator().calibrate(video_path)
        return

    calibration = TableCalibrator().calibrate(video_path)

    assert calibration.field is not None
    if expects_supported_layout:
        assert len(calibration.rods) == 8
        assert calibration.warnings == ()
    else:
        assert calibration.rods == ()
        assert calibration.warnings
        assert calibration.warnings[0].startswith("Rod consensus requires review:")