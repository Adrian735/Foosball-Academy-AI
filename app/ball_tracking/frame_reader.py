"""Sequential full-video decoding for standalone ball tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from app.detection.contracts.video_contracts import VideoMetadata


class BallVideoReadError(ValueError):
    """Raised when a video cannot provide valid sequential tracking input."""


@dataclass(frozen=True)
class SequentialFrame:
    """One decoded BGR frame at its original source position."""

    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


@dataclass(frozen=True)
class SequentialFrameReadResult:
    """Video metadata, decoded frames, and explicit decoder diagnostics."""

    metadata: VideoMetadata
    frames: Tuple[SequentialFrame, ...]
    warnings: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Return whether decoding reached the advertised frame count."""
        return not self.warnings


class SequentialFrameReader:
    """Decode every video frame in source order without sampling or seeking."""

    def read(self, video_path: str) -> SequentialFrameReadResult:
        """Read a complete video sequentially and preserve source indices.

        Raises:
            BallVideoReadError: If the video cannot be opened or has incomplete
                technical metadata.
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise BallVideoReadError(f"Unable to open video: {video_path}")

        try:
            metadata = self._read_metadata(video_path, capture)
            frames: list[SequentialFrame] = []
            warnings: list[str] = []
            expected_frame_count = metadata.frame_count
            frame_index = 0
            while True:
                was_read, image = capture.read()
                if not was_read or image is None:
                    if frame_index < expected_frame_count:
                        warnings.append(
                            f"Decoder stopped at source frame {frame_index}; "
                            f"expected {expected_frame_count} frames"
                        )
                    break
                timestamp_seconds = frame_index / metadata.frames_per_second
                frames.append(SequentialFrame(frame_index, timestamp_seconds, image))
                frame_index += 1

            if frame_index > expected_frame_count:
                warnings.append(
                    f"Decoder produced {frame_index} frames; "
                    f"metadata advertised {expected_frame_count}"
                )
            return SequentialFrameReadResult(metadata, tuple(frames), tuple(warnings))
        finally:
            capture.release()

    def iter_frames(self, video_path: str) -> Iterator[SequentialFrame]:
        """Yield decoded frames in source order while releasing the decoder."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise BallVideoReadError(f"Unable to open video: {video_path}")
        try:
            metadata = self._read_metadata(video_path, capture)
            for frame_index in range(metadata.frame_count):
                was_read, image = capture.read()
                if not was_read or image is None:
                    raise BallVideoReadError(
                        f"Decoder stopped at source frame {frame_index}; "
                        f"expected {metadata.frame_count} frames"
                    )
                yield SequentialFrame(frame_index, frame_index / metadata.frames_per_second, image)
        finally:
            capture.release()

    @staticmethod
    def _read_metadata(video_path: str, capture: cv2.VideoCapture) -> VideoMetadata:
        """Read and validate technical properties from an open decoder."""
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise BallVideoReadError(f"Video metadata is incomplete: {video_path}")
        return VideoMetadata(
            path=str(Path(video_path)),
            frames_per_second=frames_per_second,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / frames_per_second,
        )
