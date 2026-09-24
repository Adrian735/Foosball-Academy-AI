"""Focused tests for temporal ball association."""

from dataclasses import replace

import numpy as np

from app.ball_tracking.config import DEFAULT_BALL_TRACKING_CONFIG
from app.ball_tracking.contracts import BallCandidate, ObservationState
from app.ball_tracking.detector import BallDetectionFrame
from app.ball_tracking.frame_reader import SequentialFrame
from app.ball_tracking.tracker import BallTracker
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.contracts.video_contracts import VideoMetadata


FIELD = FieldGeometry(
    corners=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    bounding_box=(0, 0, 100, 100),
    confidence=1.0,
    detection_method="synthetic",
)
METADATA = VideoMetadata("synthetic.mp4", 10.0, 0, 100, 100, 0.0)


def _candidate(x: float, confidence: float = 0.9) -> BallCandidate:
    """Build a candidate directly in the synthetic canonical plane."""
    return BallCandidate(
        center=(x, 50.0),
        canonical_center=(x * 10.0, 300.0),
        bounding_box=(x - 2.0, 48.0, 4.0, 4.0),
        area=12.0,
        circularity=0.9,
        aspect_ratio=1.0,
        confidence=confidence,
    )


class _ScriptedDetector:
    """Return candidates keyed by source frame index."""

    def __init__(self, candidates: dict[int, tuple[BallCandidate, ...]]) -> None:
        self._candidates = candidates

    def detect(self, frame: np.ndarray, field_geometry: FieldGeometry) -> BallDetectionFrame:
        """Return the scripted candidates for the frame encoded in its pixel."""
        frame_index = int(frame[0, 0, 0])
        return BallDetectionFrame(self._candidates.get(frame_index, ()), ())


def _frames(count: int) -> tuple[SequentialFrame, ...]:
    """Build timestamped synthetic frames at ten frames per second."""
    return tuple(
        SequentialFrame(index, index / 10.0, np.full((1, 1, 3), index, dtype=np.uint8))
        for index in range(count)
    )


def _tracker(candidates: dict[int, tuple[BallCandidate, ...]]) -> BallTracker:
    """Create a tracker with a scripted stateless detector."""
    config = replace(
        DEFAULT_BALL_TRACKING_CONFIG,
        maximum_displacement_canonical_per_second=100.0,
        minimum_track_coverage=0.0,
        maximum_unresolved_gap_seconds=0.75,
    )
    return BallTracker(_ScriptedDetector(candidates), config)


def test_tracker_associates_continuous_motion() -> None:
    """Continuous candidates produce one detected observation per frame."""
    result = _tracker({index: (_candidate(index),) for index in range(4)}).track(
        _frames(4), FIELD, METADATA, "calibration-1"
    )

    assert [observation.state for observation in result.track.observations] == [
        ObservationState.DETECTED
    ] * 4
    assert result.track.detection_coverage == 1.0


def test_tracker_recovers_after_short_occlusion() -> None:
    """A brief missing interval is preserved and later detection recovers."""
    result = _tracker({0: (_candidate(10.0),), 2: (_candidate(11.0),)}).track(
        _frames(3), FIELD, METADATA, "calibration-1"
    )

    assert [observation.state for observation in result.track.observations] == [
        ObservationState.DETECTED,
        ObservationState.MISSED,
        ObservationState.DETECTED,
    ]
    assert result.track.longest_missing_interval_seconds == 0.1


def test_tracker_reports_long_unresolved_gap() -> None:
    """A long gap does not reconnect a candidate to stale history."""
    result = _tracker({0: (_candidate(10.0),), 9: (_candidate(11.0),)}).track(
        _frames(10), FIELD, METADATA, "calibration-1"
    )

    assert all(
        observation.state is ObservationState.MISSED
        for observation in result.track.observations[1:9]
    )
    assert result.track.observations[9].state is ObservationState.UNCERTAIN
    assert result.track.observations[9].diagnostics == ("recovery_gap_exceeded",)
    assert result.track.longest_missing_interval_seconds == 0.8
    assert "unresolved_missing_interval" in result.track.warnings


def test_tracker_prefers_competing_candidate_with_continuity() -> None:
    """A high-quality distractor does not displace the nearby candidate."""
    result = _tracker(
        {
            0: (_candidate(10.0),),
            1: (_candidate(11.0, 0.7), _candidate(18.0, 0.99)),
        }
    ).track(_frames(2), FIELD, METADATA, "calibration-1")

    assert result.track.observations[1].candidate is not None
    assert result.track.observations[1].candidate.canonical_center == (110.0, 300.0)


def test_tracker_marks_implausible_jump_uncertain() -> None:
    """An implausible jump is uncertain and does not update the trajectory."""
    result = _tracker({0: (_candidate(10.0),), 1: (_candidate(80.0),), 2: (_candidate(11.0),)}).track(
        _frames(3), FIELD, METADATA, "calibration-1"
    )

    assert result.track.observations[1].state is ObservationState.UNCERTAIN
    assert result.track.observations[1].candidate is None
    assert result.track.observations[2].state is ObservationState.DETECTED
    assert result.track.observations[2].candidate is not None
    assert result.track.observations[2].candidate.canonical_center == (110.0, 300.0)