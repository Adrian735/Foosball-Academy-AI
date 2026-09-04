"""Tests for startup-window video sampling and quality gates."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.detection.config import DetectionConfig
from app.detection.startup_frames import FrameQualityEvaluator, StartupFrameReader, VideoValidationError


def _write_video(path: Path, frames: list[np.ndarray], frames_per_second: float = 5.0) -> None:
    """Write BGR frames as a small Motion JPEG test video."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), frames_per_second, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _checkerboard_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Return a bright, sharp BGR frame suitable for quality-gate tests."""
    tile_size = 16
    rows, columns = np.indices((height, width))
    board = ((rows // tile_size + columns // tile_size) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def test_frame_quality_rejects_dark_and_blurred_images() -> None:
    """The evaluator explains dark and low-detail frame rejections."""
    evaluator = FrameQualityEvaluator()
    dark_quality = evaluator.evaluate(np.zeros((720, 1280, 3), dtype=np.uint8), 0, 0.0)
    blurred_quality = evaluator.evaluate(np.full((720, 1280, 3), 128, dtype=np.uint8), 1, 0.2)

    assert dark_quality.rejection_reason == "too_dark"
    assert blurred_quality.rejection_reason == "too_blurry"


def test_reader_returns_evenly_spaced_accepted_frames(tmp_path: Path) -> None:
    """A valid video yields configured samples from only its startup window."""
    video_path = tmp_path / "valid.avi"
    _write_video(video_path, [_checkerboard_frame() for _ in range(25)])

    result = StartupFrameReader().read(str(video_path))

    assert result.metadata.duration_seconds >= 5.0
    assert len(result.frame_quality) == 15
    assert len(result.accepted_frames) == 15
    assert result.accepted_frames[0].frame_index == 0
    assert result.accepted_frames[-1].frame_index == 14
    assert all(frame.quality.accepted for frame in result.accepted_frames)


@pytest.mark.parametrize(
    ("frame_count", "width", "height", "error_message"),
    [
        (20, 1280, 720, "duration"),
        (25, 640, 480, "resolution"),
    ],
)
def test_reader_rejects_videos_outside_supported_input_contract(
    tmp_path: Path,
    frame_count: int,
    width: int,
    height: int,
    error_message: str,
) -> None:
    """Too-short and low-resolution videos fail before calibration sampling."""
    video_path = tmp_path / "unsupported.avi"
    _write_video(video_path, [_checkerboard_frame(width, height) for _ in range(frame_count)])

    with pytest.raises(VideoValidationError, match=error_message):
        StartupFrameReader().read(str(video_path))


def test_reader_rejects_an_unreadable_path(tmp_path: Path) -> None:
    """A nonexistent video reports a structured validation error."""
    with pytest.raises(VideoValidationError, match="Unable to open"):
        StartupFrameReader().read(str(tmp_path / "missing.mp4"))