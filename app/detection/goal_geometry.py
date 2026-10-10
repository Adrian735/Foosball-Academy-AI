"""Estimate Bonzini B90 goal geometry from calibrated field dimensions."""

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import FieldGeometry, GoalEnd, GoalMouth


class BonziniGoalGeometryEstimator:
    """Scale standard B90 goal dimensions into canonical field coordinates."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create an estimator using the versioned Bonzini geometry settings."""
        self._config = config

    def estimate(self, field: FieldGeometry) -> tuple[GoalMouth, ...]:
        """Estimate both centered goal mouths using detected field confidence."""
        width = self._config.canonical_field_width
        height = self._config.canonical_field_height
        opening_width = (
            self._config.bonzini_goal_opening_width_mm
            / self._config.bonzini_playfield_width_mm
            * width
        )
        mouth_depth = (
            self._config.bonzini_goal_mouth_depth_mm
            / self._config.bonzini_playfield_length_mm
            * height
        )
        center_x = (width - 1) / 2.0
        left = center_x - opening_width / 2.0
        right = center_x + opening_width / 2.0
        diagnostics = ("bonzini_b90_geometry_estimate",)
        return (
            GoalMouth(
                end=GoalEnd.START,
                opening_bounds=(left, -mouth_depth, right, 0.0),
                crossing_line_y=0.0,
                confidence=field.confidence,
                diagnostics=diagnostics,
            ),
            GoalMouth(
                end=GoalEnd.END,
                opening_bounds=(left, height - 1.0, right, height - 1.0 + mouth_depth),
                crossing_line_y=float(height - 1),
                confidence=field.confidence,
                diagnostics=diagnostics,
            ),
        )
