"""Startup-window video sampling for static table calibration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.video_contracts import FrameQuality, SampledFrame, VideoMetadata


class VideoValidationError(ValueError):
    """Raised when a video cannot provide a supported calibration sample."""


@dataclass(frozen=True)
class StartupFrameReadResult:
    """Metadata, all quality diagnostics, and accepted startup frames.

    Rejected frames are represented only in ``frame_quality`` to avoid passing
    unusable image buffers to later calibration stages.
    """

    metadata: VideoMetadata
    frame_quality: Tuple[FrameQuality, ...]
    accepted_frames: Tuple[SampledFrame, ...]


class FrameQualityEvaluator:
    """Evaluate sampled BGR frames against brightness and blur thresholds."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create an evaluator using the supplied immutable detector config."""
        self._config = config

    def evaluate(self, image: np.ndarray, frame_index: int, timestamp_seconds: float) -> FrameQuality:
        """Return quality diagnostics for one BGR image and its video location."""
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_score = float(cv2.mean(grayscale_image)[0])
        blur_score = float(cv2.Laplacian(grayscale_image, cv2.CV_64F).var())

        if brightness_score < self._config.minimum_mean_luminance:
            return FrameQuality(
                frame_index,
                timestamp_seconds,
                blur_score,
                brightness_score,
                False,
                "too_dark",
            )
        if blur_score < self._config.minimum_laplacian_variance:
            return FrameQuality(
                frame_index,
                timestamp_seconds,
                blur_score,
                brightness_score,
                False,
                "too_blurry",
            )
        return FrameQuality(frame_index, timestamp_seconds, blur_score, brightness_score, True)


class StartupFrameReader:
    """Read quality-screened BGR samples from a video's calibration window."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a reader using the supplied immutable detector config."""
        self._config = config
        self._quality_evaluator = FrameQualityEvaluator(config)

    def read(self, video_path: str) -> StartupFrameReadResult:
        """Validate ``video_path`` and return evenly spaced accepted startup frames.

        Raises:
            VideoValidationError: If the video is unreadable, unsupported, or
                does not meet the configured duration and resolution minimums.
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise VideoValidationError(f"Unable to open video: {video_path}")

        try:
            metadata = self._read_metadata(video_path, capture)
            self._validate_metadata(metadata)
            frame_indices = self._sample_frame_indices(metadata)
            return self._read_samples(capture, metadata, frame_indices)
        finally:
            capture.release()

    def _read_metadata(self, video_path: str, capture: cv2.VideoCapture) -> VideoMetadata:
        """Read technical properties from an open video capture."""
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise VideoValidationError(f"Video metadata is incomplete: {video_path}")
        return VideoMetadata(
            path=str(Path(video_path)),
            frames_per_second=frames_per_second,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / frames_per_second,
        )

    def _validate_metadata(self, metadata: VideoMetadata) -> None:
        """Reject video metadata outside the supported Bonzini input contract."""
        if metadata.duration_seconds < self._config.minimum_video_duration_seconds:
            raise VideoValidationError("Video is shorter than the minimum supported duration")
        if (
            metadata.width < self._config.minimum_video_width
            or metadata.height < self._config.minimum_video_height
        ):
            raise VideoValidationError("Video resolution is below the supported minimum")

    def _sample_frame_indices(self, metadata: VideoMetadata) -> Tuple[int, ...]:
        """Return unique frame indices distributed through the configured window."""
        startup_frame_count = min(
            metadata.frame_count,
            self._config.maximum_calibration_frames,
            max(1, int(metadata.frames_per_second * self._config.calibration_window_seconds)),
        )
        sample_count = min(self._config.sampled_frame_count, startup_frame_count)
        return tuple(
            int(frame_index)
            for frame_index in np.linspace(0, startup_frame_count - 1, sample_count, dtype=int)
        )

    def _read_samples(
        self,
        capture: cv2.VideoCapture,
        metadata: VideoMetadata,
        frame_indices: Tuple[int, ...],
    ) -> StartupFrameReadResult:
        """Decode selected frames and separate accepted images from diagnostics."""
        qualities: list[FrameQuality] = []
        accepted_frames: list[SampledFrame] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            was_read, image = capture.read()
            timestamp_seconds = frame_index / metadata.frames_per_second
            if not was_read or image is None:
                qualities.append(
                    FrameQuality(frame_index, timestamp_seconds, 0.0, 0.0, False, "unreadable_frame")
                )
                continue

            quality = self._quality_evaluator.evaluate(image, frame_index, timestamp_seconds)
            qualities.append(quality)
            if quality.accepted:
                accepted_frames.append(SampledFrame(frame_index, timestamp_seconds, image, quality))

        return StartupFrameReadResult(metadata, tuple(qualities), tuple(accepted_frames))