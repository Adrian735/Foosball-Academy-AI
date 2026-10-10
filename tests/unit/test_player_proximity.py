"""Tests for measured player-position proximity events."""

import pytest

from app.ball_tracking.contracts import BallCandidate, BallObservation, BallTrack, ObservationState
from app.detection.contracts.table_contracts import GoalEnd, GoalMouth
from app.event_analysis import EventAnalyzer, EventType
from app.event_analysis.config import DEFAULT_EVENT_ANALYSIS_CONFIG
from app.event_analysis.contracts import (
    PlayerPositionObservation,
    PlayerPositionState,
    PlayerPositionTrack,
)

_GOAL_MOUTHS = (
    GoalMouth(GoalEnd.START, (350.0, -75.0, 650.0, 0.0), 0.0, 0.9),
    GoalMouth(GoalEnd.END, (350.0, 599.0, 650.0, 674.0), 599.0, 0.9),
)


def _ball_observation(frame: int, position: tuple[float, float]) -> BallObservation:
    """Create a detected ball observation at one canonical position."""
    candidate = BallCandidate(
        center=position,
        canonical_center=position,
        bounding_box=(position[0] - 2, position[1] - 2, 4.0, 4.0),
        area=12.0,
        circularity=0.9,
        aspect_ratio=1.0,
        confidence=0.9,
    )
    return BallObservation(
        frame_index=frame,
        timestamp_seconds=frame / 10.0,
        state=ObservationState.DETECTED,
        candidate=candidate,
        confidence=0.9,
    )


def _player_observation(
    frame: int,
    position: tuple[float, float] | None = (500.0, 300.0),
    *,
    state: PlayerPositionState = PlayerPositionState.DETECTED,
    confidence: float = 0.9,
) -> PlayerPositionObservation:
    """Create a measured or explicitly unavailable player observation."""
    return PlayerPositionObservation(
        frame_index=frame,
        timestamp_seconds=frame / 10.0,
        state=state,
        canonical_position=position,
        confidence=confidence,
    )


def _ball_track(positions: tuple[tuple[float, float], ...]) -> BallTrack:
    """Build a detected ball track with positions indexed from frame zero."""
    observations = tuple(
        _ball_observation(frame, position)
        for frame, position in enumerate(positions)
    )
    return BallTrack({}, "calibration-1", "tracking-1", observations, 1.0, 0.0, 0.9)


def _player_track(
    observations: tuple[PlayerPositionObservation, ...],
) -> PlayerPositionTrack:
    """Build a track for the selected forward-middle player."""
    return PlayerPositionTrack("blue-3bar-middle", observations, 0.8)


def test_emits_proximity_entry_and_exit_with_physical_distance_and_duration() -> None:
    """Adjacent measured points produce balanced events and retained evidence."""
    balls = _ball_track(((500.0, 200.0), (500.0, 270.0), (500.0, 280.0), (500.0, 250.0)))
    players = _player_track(tuple(_player_observation(frame) for frame in range(4)))

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert [event.event_type for event in result.events] == [
        EventType.BALL_ENTERED_PROXIMITY,
        EventType.BALL_LEFT_PROXIMITY,
    ]
    entered, left = result.events
    assert entered.evidence[1].canonical_position == (500.0, 270.0)
    assert entered.evidence[1].canonical_player_position == (500.0, 300.0)
    assert entered.details == (
        ("player_id", "blue-3bar-middle"),
        ("distance_mm", 60.0),
        ("proximity_radius_mm", 75.0),
    )
    assert dict(left.details) == {
        "player_id": "blue-3bar-middle",
        "duration_seconds": pytest.approx(0.2),
        "distance_mm": 100.0,
        "proximity_radius_mm": 75.0,
    }
    assert dict(entered.details) == {
        "player_id": "blue-3bar-middle",
        "distance_mm": 60.0,
        "proximity_radius_mm": 75.0,
    }
    assert left.confidence == 0.8
    assert result.warnings == ()


def test_unmatched_source_frame_does_not_bridge_a_proximity_interval() -> None:
    """A gap after entry warns and never fabricates the missing leave boundary."""
    balls = _ball_track(
        ((500.0, 200.0), (500.0, 270.0), (500.0, 280.0), (500.0, 250.0))
    )
    players = _player_track(
        (
            _player_observation(0),
            _player_observation(1),
            _player_observation(3),
        )
    )

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert [event.event_type for event in result.events] == [
        EventType.BALL_ENTERED_PROXIMITY,
    ]
    assert "player_proximity_interval_incomplete" in result.warnings


def test_trailing_proximity_interval_has_no_fabricated_leave_event() -> None:
    """An active interval at track end is reported unresolved, not closed."""
    balls = _ball_track(((500.0, 200.0), (500.0, 270.0), (500.0, 280.0)))
    players = _player_track(tuple(_player_observation(frame) for frame in range(3)))

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert [event.event_type for event in result.events] == [
        EventType.BALL_ENTERED_PROXIMITY,
    ]
    assert "player_proximity_interval_unresolved" in result.warnings


def test_missing_player_track_is_reported_as_unavailable() -> None:
    """No player input cannot be interpreted as no proximity event."""
    result = EventAnalyzer().analyze(
        _ball_track(((500.0, 200.0),)),
        goal_mouths=_GOAL_MOUTHS,
    )

    assert "player_position_track_unavailable" in result.warnings
    assert result.events == ()


def test_low_confidence_player_observations_are_not_used() -> None:
    """Ambiguous player detections suppress transitions and report uncertainty."""
    balls = _ball_track(((500.0, 200.0), (500.0, 270.0)))
    players = _player_track(
        (
            _player_observation(0),
            _player_observation(1, confidence=0.1),
        )
    )

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert not any(
        event.event_type in (
            EventType.BALL_ENTERED_PROXIMITY,
            EventType.BALL_LEFT_PROXIMITY,
        )
        for event in result.events
    )
    assert "player_proximity_evidence_unavailable" in result.warnings


def test_low_confidence_player_track_is_not_used() -> None:
    """Aggregate track uncertainty gates proximity even if one frame is strong."""
    balls = _ball_track(((500.0, 200.0), (500.0, 270.0)))
    players = PlayerPositionTrack(
        "blue-3bar-middle",
        tuple(_player_observation(frame) for frame in range(2)),
        0.2,
    )

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert not any(
        event.event_type in (
            EventType.BALL_ENTERED_PROXIMITY,
            EventType.BALL_LEFT_PROXIMITY,
        )
        for event in result.events
    )
    assert "player_proximity_evidence_unavailable" in result.warnings


def test_initial_inside_state_does_not_invent_an_entry_event() -> None:
    """Proximity already active at the first paired frame has no asserted entry."""
    balls = _ball_track(((500.0, 270.0), (500.0, 250.0)))
    players = _player_track(tuple(_player_observation(frame) for frame in range(2)))

    result = EventAnalyzer().analyze(balls, goal_mouths=_GOAL_MOUTHS, player_track=players)

    assert not any(
        event.event_type is EventType.BALL_ENTERED_PROXIMITY
        for event in result.events
    )
    assert "player_proximity_state_unknown_at_start" in result.warnings


def test_player_contract_round_trips_as_json_safe_values() -> None:
    """Player track contracts retain measured coordinates and observation states."""
    import json

    track = _player_track(
        (
            _player_observation(0),
            _player_observation(
                1,
                None,
                state=PlayerPositionState.UNCERTAIN,
                confidence=0.2,
            ),
        )
    )

    payload = track.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["observations"][1]["state"] == "uncertain"
    assert payload["observations"][1]["canonical_position"] is None


@pytest.mark.parametrize(
    ("state", "position"),
    (
        (PlayerPositionState.DETECTED, None),
        (PlayerPositionState.MISSED, (500.0, 300.0)),
    ),
)
def test_player_contract_rejects_state_position_mismatch(
    state: PlayerPositionState,
    position: tuple[float, float] | None,
) -> None:
    """Only detected player observations can carry a measured position."""
    with pytest.raises(ValueError, match="canonical_position"):
        _player_observation(0, position, state=state)


def test_proximity_radius_is_versioned_and_defaults_to_seventy_five_mm() -> None:
    """The agreed threshold is explicit in the immutable event configuration."""
    assert DEFAULT_EVENT_ANALYSIS_CONFIG.player_proximity_radius_mm == 75.0
