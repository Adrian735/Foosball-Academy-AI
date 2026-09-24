"""Immutable thresholds for standalone ball tracking."""

from dataclasses import dataclass
from typing import Tuple

HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass(frozen=True)
class BallTrackingConfig:
    """Versioned ball detector and track-association thresholds."""

    config_version: str = "1"
    yellow_hsv_range: HSVRange = ((18, 80, 80), (32, 255, 255))
    recovery_yellow_hsv_range: HSVRange = ((16, 60, 60), (34, 255, 255))
    morphology_kernel_size: int = 5
    canonical_field_width: int = 1000
    canonical_field_height: int = 600
    minimum_contour_area_ratio: float = 0.00002
    maximum_contour_area_ratio: float = 0.02
    minimum_circularity: float = 0.45
    maximum_aspect_ratio: float = 2.5
    minimum_aspect_ratio: float = 0.4
    field_inset_pixels: int = 10
    field_edge_margin_canonical: float = 12.0
    maximum_displacement_canonical_per_second: float = 1800.0
    maximum_recovery_gap_seconds: float = 0.35
    circularity_score_weight: float = 0.7
    continuity_score_weight: float = 0.3
    minimum_detection_confidence: float = 0.45
    minimum_track_coverage: float = 0.60
    maximum_unresolved_gap_seconds: float = 0.75


DEFAULT_BALL_TRACKING_CONFIG = BallTrackingConfig()
