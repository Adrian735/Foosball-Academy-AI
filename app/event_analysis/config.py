"""Versioned thresholds for deterministic event analysis."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EventAnalysisConfig:
    """Versioned thresholds for goal and measured-player proximity events."""

    config_version: str = "3"
    minimum_goal_mouth_confidence: float = 0.5
    goal_rearm_distance_canonical: float = 20.0
    player_proximity_radius_mm: float = 75.0
    minimum_player_position_confidence: float = 0.5
    bonzini_playfield_width_mm: float = 700.0
    bonzini_playfield_length_mm: float = 1200.0
    canonical_field_width: int = 1000
    canonical_field_height: int = 600
    maximum_aligned_timestamp_difference_seconds: float = 0.001

    def __post_init__(self) -> None:
        """Reject invalid versions, confidence, dimensions, and proximity limits."""
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ValueError("config_version must be a non-empty string")
        confidence = float(self.minimum_goal_mouth_confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("minimum_goal_mouth_confidence must be finite and within [0, 1]")
        object.__setattr__(self, "minimum_goal_mouth_confidence", confidence)
        distance = float(self.goal_rearm_distance_canonical)
        if not isfinite(distance) or distance < 0.0:
            raise ValueError("goal_rearm_distance_canonical must be finite and non-negative")
        object.__setattr__(self, "goal_rearm_distance_canonical", distance)
        proximity_radius = float(self.player_proximity_radius_mm)
        if not isfinite(proximity_radius) or proximity_radius < 0.0:
            raise ValueError("player_proximity_radius_mm must be finite and non-negative")
        object.__setattr__(self, "player_proximity_radius_mm", proximity_radius)
        player_confidence = float(self.minimum_player_position_confidence)
        if not isfinite(player_confidence) or not 0.0 <= player_confidence <= 1.0:
            raise ValueError("minimum_player_position_confidence must be finite and within [0, 1]")
        object.__setattr__(self, "minimum_player_position_confidence", player_confidence)
        for field_name in ("bonzini_playfield_width_mm", "bonzini_playfield_length_mm"):
            dimension = float(getattr(self, field_name))
            if not isfinite(dimension) or dimension <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, dimension)
        for field_name in ("canonical_field_width", "canonical_field_height"):
            dimension = getattr(self, field_name)
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        timestamp_delta = float(self.maximum_aligned_timestamp_difference_seconds)
        if not isfinite(timestamp_delta) or timestamp_delta < 0.0:
            raise ValueError(
                "maximum_aligned_timestamp_difference_seconds must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "maximum_aligned_timestamp_difference_seconds",
            timestamp_delta,
        )


DEFAULT_EVENT_ANALYSIS_CONFIG = EventAnalysisConfig()
