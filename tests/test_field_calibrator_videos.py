"""Regression checks for supplied field-detection video fixtures."""

from pathlib import Path

from app.detection.calibrator import TableCalibrator


VIDEO_DIRECTORY = Path(__file__).parent / "table-detection_tests"


def test_video_6_produces_stable_field_calibration() -> None:
    """Video 6 has a stable field and should calibrate without review warnings."""
    calibration = TableCalibrator().calibrate_field(
        str(VIDEO_DIRECTORY / "table-detection_test-6.mp4")
    )

    assert calibration.field is not None
    assert calibration.confidence >= 0.50
    assert calibration.warnings == ()
    assert calibration.detector_config_version == "2"