"""Backward-compatible facade for detection contracts.

New code should import contracts from the responsibility-specific modules.
"""

from app.detection.contracts.rod_contracts import LineSegment, Point, Rod, RodCandidate
from app.detection.contracts.table_contracts import Corners, FieldGeometry, TableCalibration
from app.detection.contracts.video_contracts import FrameQuality, SampledFrame, VideoMetadata

__all__ = [
    "Corners",
    "FieldGeometry",
    "FrameQuality",
    "LineSegment",
    "Point",
    "Rod",
    "RodCandidate",
    "SampledFrame",
    "TableCalibration",
    "VideoMetadata",
]
