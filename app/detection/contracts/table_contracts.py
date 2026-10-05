"""Serializable contracts for field geometry and table calibration."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Optional, Tuple

from app.detection.contracts.rod_contracts import Rod
from app.detection.contracts.video_contracts import FrameQuality, VideoMetadata

Point = Tuple[float, float]
Corners = Tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class FieldGeometry:
    """Detected playable-field polygon, bounds, and detection confidence."""

    corners: Corners
    bounding_box: Tuple[int, int, int, int]
    confidence: float
    detection_method: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe field geometry for diagnostics and persistence."""
        return {
            "corners": [list(point) for point in self.corners],
            "bounding_box": list(self.bounding_box),
            "confidence": self.confidence,
            "detection_method": self.detection_method,
        }


class GoalEnd(str, Enum):
    """Canonical end of the playable field where a goal mouth is visible."""

    START = "start"
    END = "end"


@dataclass(frozen=True)
class GoalMouth:
    """Visually detected goal aperture and its canonical crossing plane."""

    end: GoalEnd
    opening_bounds: Tuple[float, float, float, float]
    crossing_line_y: float
    confidence: float
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate finite, ordered aperture bounds and confidence."""
        if not isinstance(self.end, GoalEnd):
            raise ValueError("end must be a GoalEnd")
        try:
            bounds = tuple(float(value) for value in self.opening_bounds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("opening_bounds must contain four finite coordinates") from error
        if len(bounds) != 4 or not all(isfinite(value) for value in bounds):
            raise ValueError("opening_bounds must contain four finite coordinates")
        left, top, right, bottom = bounds
        if left >= right or top >= bottom:
            raise ValueError("opening_bounds must be ordered left, top, right, bottom")
        object.__setattr__(self, "opening_bounds", bounds)

        crossing_line = float(self.crossing_line_y)
        if not isfinite(crossing_line):
            raise ValueError("crossing_line_y must be finite")
        object.__setattr__(self, "crossing_line_y", crossing_line)

        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)

        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, str) for value in diagnostics):
            raise ValueError("diagnostics must contain strings")
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical geometry as plain JSON-safe values."""
        return {
            "end": self.end.value,
            "opening_bounds": list(self.opening_bounds),
            "crossing_line_y": self.crossing_line_y,
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class TableCalibration:
    """Complete static-table calibration consumed by later CV pipeline stages."""

    metadata: VideoMetadata
    sampled_frame_quality: Tuple[FrameQuality, ...]
    field: Optional[FieldGeometry]
    rods: Tuple[Rod, ...]
    confidence: float
    warnings: Tuple[str, ...] = ()
    detector_config_version: str = "1"
    goal_mouths: Tuple[GoalMouth, ...] = ()
    goal_warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe calibration report."""
        return {
            "metadata": self.metadata.to_dict(),
            "sampled_frame_quality": [quality.to_dict() for quality in self.sampled_frame_quality],
            "field": self.field.to_dict() if self.field else None,
            "rods": [rod.to_dict() for rod in self.rods],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "detector_config_version": self.detector_config_version,
            "goal_mouths": [mouth.to_dict() for mouth in self.goal_mouths],
            "goal_warnings": list(self.goal_warnings),
        }