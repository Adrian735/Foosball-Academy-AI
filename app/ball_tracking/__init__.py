"""Standalone ball-tracking contracts and configuration."""

from app.ball_tracking.calibration import CalibrationLoadError, load_table_calibration, require_trackable_calibration
from app.ball_tracking.config import BallTrackingConfig, DEFAULT_BALL_TRACKING_CONFIG
from app.ball_tracking.contracts import (
    BallCandidate,
    BallObservation,
    BallTrack,
    BallTrackingResult,
    ObservationState,
)
from app.ball_tracking.detector import BallDetectionFrame, BallDetector
from app.ball_tracking.frame_reader import (
    BallVideoReadError,
    SequentialFrame,
    SequentialFrameReadResult,
    SequentialFrameReader,
)
from app.ball_tracking.tracker import BallTracker

__all__ = [
    "BallCandidate",
    "BallDetectionFrame",
    "BallDetector",
    "BallObservation",
    "BallTrack",
    "BallTrackingResult",
    "BallTracker",
    "BallTrackingConfig",
    "BallVideoReadError",
    "CalibrationLoadError",
    "DEFAULT_BALL_TRACKING_CONFIG",
    "ObservationState",
    "SequentialFrame",
    "SequentialFrameReadResult",
    "SequentialFrameReader",
    "load_table_calibration",
    "require_trackable_calibration",
]
