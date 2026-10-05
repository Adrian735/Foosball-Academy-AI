"""Tests for ball lost/reacquired event extraction."""

import pytest

from app.ball_tracking.contracts import (
    BallCandidate,
    BallObservation,
    BallTrack,
    ObservationState,
)
from app.event_analysis.analyzer import EventAnalyzer
from app.event_analysis.contracts import EventType


def _observation(
    frame: int,
    state: ObservationState,
    *,
    confidence: float = 0.8,
    diagnostics: tuple[str, ...] = (),
) -> BallObservation:
    """Create a timestamped synthetic observation with a position only when detected."""
    candidate = None
    if state is ObservationState.DETECTED:
        candidate = BallCandidate(
            center=(float(frame), 20.0),
            canonical_center=(float(frame * 10), 200.0),
            bounding_box=(float(frame), 20.0, 4.0, 4.0),
            area=12.0,
            circularity=0.9,
            aspect_ratio=1.0,
            confidence=confidence,
        )
    return BallObservation(
        frame_index=frame,
        timestamp_seconds=frame / 10.0,
        state=state,
        candidate=candidate,
        confidence=confidence if state is ObservationState.DETECTED else 0.0,
        diagnostics=diagnostics,
    )


def _track(
    observations: tuple[BallObservation, ...],
    *,
    confidence: float = 0.9,
    warnings: tuple[str, ...] = (),
) -> BallTrack:
    """Create a synthetic ball track for state-transition tests."""
    return BallTrack(
        video_metadata={},
        calibration_config_version="calibration-1",
        tracking_config_version="tracking-1",
        observations=observations,
        detection_coverage=0.7,
        longest_missing_interval_seconds=0.2,
        confidence=confidence,
        warnings=warnings,
    )


def test_detects_one_loss_and_reacquisition_while_preserving_gap_evidence() -> None:
    """Missed and uncertain frames remain explicit and are never interpolated."""
    observations = (
        _observation(0, ObservationState.DETECTED, confidence=0.9),
        _observation(1, ObservationState.MISSED, diagnostics=("no_candidate",)),
        _observation(2, ObservationState.UNCERTAIN, diagnostics=("implausible_jump",)),
        _observation(3, ObservationState.MISSED),
        _observation(4, ObservationState.DETECTED, confidence=0.8),
    )

    result = EventAnalyzer().analyze(_track(observations))

    assert [event.event_type for event in result.events] == [
        EventType.BALL_LOST,
        EventType.BALL_REACQUIRED,
    ]
    lost, reacquired = result.events
    assert [(item.frame_index, item.timestamp_seconds) for item in lost.evidence] == [
        (0, 0.0),
        (1, 0.1),
    ]
    assert [item.observation_state for item in reacquired.evidence] == [
        ObservationState.DETECTED,
        ObservationState.MISSED,
        ObservationState.UNCERTAIN,
        ObservationState.MISSED,
        ObservationState.DETECTED,
    ]
    assert [item.canonical_position for item in reacquired.evidence] == [
        (0.0, 200.0),
        None,
        None,
        None,
        (40.0, 200.0),
    ]
    assert reacquired.evidence[2].to_dict()["timestamp_seconds"] == 0.2
    assert lost.confidence == 0.0
    assert reacquired.confidence == 0.8
    assert result.warnings == ()


def test_consecutive_non_detected_frames_emit_one_loss_event() -> None:
    """A single unresolved interval does not emit repeated loss transitions."""
    track = _track(
        (
            _observation(0, ObservationState.DETECTED),
            _observation(1, ObservationState.MISSED),
            _observation(2, ObservationState.MISSED),
            _observation(3, ObservationState.UNCERTAIN),
            _observation(4, ObservationState.DETECTED),
        )
    )

    result = EventAnalyzer().analyze(track)

    assert [event.event_type for event in result.events] == [
        EventType.BALL_LOST,
        EventType.BALL_REACQUIRED,
    ]


def test_trailing_loss_emits_no_reacquisition_and_warns() -> None:
    """An unresolved end-of-track gap is explicit without an invented recovery."""
    result = EventAnalyzer().analyze(
        _track(
            (
                _observation(0, ObservationState.DETECTED),
                _observation(1, ObservationState.MISSED),
                _observation(2, ObservationState.UNCERTAIN),
            )
        )
    )

    assert [event.event_type for event in result.events] == [EventType.BALL_LOST]
    assert "unresolved_ball_loss_interval" in result.warnings


def test_leading_gap_does_not_assert_ball_was_lost() -> None:
    """No loss event is inferred before a ball has ever been detected."""
    result = EventAnalyzer().analyze(
        _track(
            (
                _observation(0, ObservationState.MISSED),
                _observation(1, ObservationState.UNCERTAIN),
                _observation(2, ObservationState.DETECTED),
            )
        )
    )

    assert result.events == ()
    assert "ball_unobserved_at_start" in result.warnings


def test_empty_track_reports_missing_observations() -> None:
    """An empty input is distinguishable from a complete zero-event track."""
    result = EventAnalyzer().analyze(_track(()))

    assert result.events == ()
    assert "no_ball_observations" in result.warnings


def test_track_warnings_are_carried_to_event_result() -> None:
    """Track-quality warnings remain available to downstream callers."""
    result = EventAnalyzer().analyze(
        _track(
            (_observation(0, ObservationState.DETECTED),),
            warnings=("track_coverage_below_threshold",),
        )
    )

    assert result.warnings == ("track_coverage_below_threshold",)


def test_analyzer_rejects_observations_out_of_source_order() -> None:
    """Transition analysis cannot silently reorder frame evidence."""
    track = _track(
        (
            _observation(2, ObservationState.DETECTED),
            _observation(1, ObservationState.MISSED),
        )
    )

    with pytest.raises(ValueError, match="source order"):
        EventAnalyzer().analyze(track)


def test_analyzer_rejects_timestamps_out_of_source_order() -> None:
    """Monotonic frame numbers do not excuse reversed source timestamps."""
    observations = (
        _observation(0, ObservationState.DETECTED),
        _observation(1, ObservationState.MISSED),
        _observation(2, ObservationState.DETECTED),
    )
    observations = (
        observations[0],
        observations[1],
        BallObservation(2, 0.05, ObservationState.DETECTED, observations[2].candidate, 0.8),
    )

    with pytest.raises(ValueError, match="source order"):
        EventAnalyzer().analyze(_track(observations))


def test_analyzer_uses_explicit_event_configuration_version() -> None:
    """Every report records the caller-provided analyzer version."""
    result = EventAnalyzer(config_version="event-v2").analyze(
        _track((_observation(0, ObservationState.DETECTED),))
    )

    assert result.config_version == "event-v2"
