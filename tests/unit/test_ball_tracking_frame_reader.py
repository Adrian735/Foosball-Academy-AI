"""Tests for sequential full-video ball-tracking frame reads."""

from pathlib import Path

import numpy as np
import pytest

from app.ball_tracking.frame_reader import BallVideoReadError, SequentialFrameReader


VIDEO_PATH = Path(__file__).parents[1] / "table-detection_tests" / "table-detection_test-1.mp4"


def test_reader_preserves_source_indices_and_monotonic_timestamps() -> None:
    """Decoded frames retain source positions and FPS-derived timestamps."""
    result = SequentialFrameReader().read(str(VIDEO_PATH))

    assert result.metadata.frame_count == 162
    assert len(result.frames) == result.metadata.frame_count
    assert [frame.frame_index for frame in result.frames[:3]] == [0, 1, 2]
    assert result.frames[0].timestamp_seconds == 0.0
    assert result.frames[1].timestamp_seconds == pytest.approx(1 / 30)
    assert all(
        earlier.timestamp_seconds < later.timestamp_seconds
        for earlier, later in zip(result.frames, result.frames[1:])
    )
    assert result.complete


def test_reader_does_not_silently_renumber_decode_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoder ending early is reported at its original source index."""
    class FakeCapture:
        def __init__(self, _: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {5: 30.0, 7: 4.0, 3: 2.0, 4: 2.0}
            return values[property_id]

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index == 2:
                return False, None
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            self.index += 1
            return True, image

        def release(self) -> None:
            pass

    monkeypatch.setattr("app.ball_tracking.frame_reader.cv2.VideoCapture", FakeCapture)

    result = SequentialFrameReader().read("fake.mp4")

    assert [frame.frame_index for frame in result.frames] == [0, 1]
    assert result.warnings == ("Decoder stopped at source frame 2; expected 4 frames",)
    assert not result.complete


def test_reader_rejects_unreadable_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unopened decoder is a clear input error."""
    class ClosedCapture:
        def __init__(self, _: str) -> None:
            pass

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr("app.ball_tracking.frame_reader.cv2.VideoCapture", ClosedCapture)

    with pytest.raises(BallVideoReadError, match="Unable to open video"):
        SequentialFrameReader().read("missing.mp4")
