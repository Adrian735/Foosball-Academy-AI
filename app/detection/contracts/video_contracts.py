"""Serializable contracts for video metadata and sampled frames."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class VideoMetadata:
    """Technical properties read from a submitted video file."""

    path: str
    frames_per_second: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for diagnostics and persistence."""
        return {
            "path": self.path,
            "frames_per_second": self.frames_per_second,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class FrameQuality:
    """Quality assessment for a sampled frame used during calibration."""

    frame_index: int
    timestamp_seconds: float
    blur_score: float
    brightness_score: float
    accepted: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe frame-quality diagnostics."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "blur_score": self.blur_score,
            "brightness_score": self.brightness_score,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class SampledFrame:
    """A startup-window frame together with its quality assessment.

    `image` remains an in-memory OpenCV array and is intentionally omitted
    from `to_dict` so persistence contracts stay JSON-safe.
    """

    frame_index: int
    timestamp_seconds: float
    image: Any
    quality: FrameQuality

    def to_dict(self) -> dict[str, Any]:
        """Return serializable frame metadata without the image buffer."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "quality": self.quality.to_dict(),
        }