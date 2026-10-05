"""Serializable inputs shared by standalone event-analysis components."""

from dataclasses import dataclass
from typing import Any

from app.ball_tracking.contracts import BallTrack
from app.detection.contracts.table_contracts import TableCalibration


@dataclass(frozen=True)
class EventAnalysisInput:
    """Pair an analyzed ball track with the static calibration that contextualizes it.

    Both values remain the existing CV contracts; event analysis does not
    duplicate their state into service or persistence models.
    """

    track: BallTrack
    calibration: TableCalibration

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe tracking and calibration reports."""
        return {
            "track": self.track.to_dict(),
            "calibration": self.calibration.to_dict(),
        }
