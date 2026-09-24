"""Immutable, JSON-safe contracts for ball tracking."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

Point = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]


class ObservationState(str, Enum):
    """Allowed per-frame ball observation states."""

    DETECTED = "detected"
    MISSED = "missed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class BallCandidate:
    """One contour candidate considered by the stateless ball detector."""

    center: Point
    canonical_center: Optional[Point]
    bounding_box: BoundingBox
    area: float
    circularity: float
    aspect_ratio: float
    confidence: float
    diagnostics: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return candidate data without OpenCV or NumPy values."""
        return {
            "center": list(self.center),
            "canonical_center": list(self.canonical_center) if self.canonical_center else None,
            "bounding_box": list(self.bounding_box),
            "area": self.area,
            "circularity": self.circularity,
            "aspect_ratio": self.aspect_ratio,
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class BallObservation:
    """One source-frame result, including explicit missing or uncertain state."""

    frame_index: int
    timestamp_seconds: float
    state: ObservationState
    candidate: Optional[BallCandidate]
    confidence: float
    diagnostics: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-safe frame observation."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "state": self.state.value,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class BallTrack:
    """Complete JSON-safe trajectory and explainable aggregate diagnostics."""

    video_metadata: dict[str, Any]
    calibration_config_version: str
    tracking_config_version: str
    observations: Tuple[BallObservation, ...]
    detection_coverage: float
    longest_missing_interval_seconds: float
    confidence: float
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted tracking report."""
        return {
            "video_metadata": self.video_metadata,
            "calibration_config_version": self.calibration_config_version,
            "tracking_config_version": self.tracking_config_version,
            "observations": [observation.to_dict() for observation in self.observations],
            "detection_coverage": self.detection_coverage,
            "longest_missing_interval_seconds": self.longest_missing_interval_seconds,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BallTrackingResult:
    """Tracking report with optional in-memory candidate diagnostics."""

    track: BallTrack
    candidates_by_frame: Tuple[Tuple[BallCandidate, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return only the JSON-safe persisted track."""
        return self.track.to_dict()
