"""Serializable contracts for field geometry and table calibration."""

from dataclasses import dataclass
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
        }