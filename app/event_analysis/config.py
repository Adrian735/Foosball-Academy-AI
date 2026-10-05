"""Versioned thresholds for deterministic event analysis."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EventAnalysisConfig:
    """Thresholds for confidence gating and goal-crossing event debouncing."""

    config_version: str = "2"
    minimum_goal_mouth_confidence: float = 0.5
    goal_rearm_distance_canonical: float = 20.0

    def __post_init__(self) -> None:
        """Reject empty versions and invalid confidence or distance thresholds."""
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


DEFAULT_EVENT_ANALYSIS_CONFIG = EventAnalysisConfig()
