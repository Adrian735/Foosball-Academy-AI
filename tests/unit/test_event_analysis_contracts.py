"""Tests for immutable, serializable event-analysis output contracts."""

import json
from dataclasses import FrozenInstanceError

import pytest

from app.ball_tracking.contracts import ObservationState
from app.event_analysis.contracts import (
    Event,
    EventAnalysisResult,
    EventEvidence,
    EventType,
    PlayerPositionObservation,
    PlayerPositionState,
    PlayerPositionTrack,
)


def _event() -> Event:
    """Create a representative event with two source-frame measurements."""
    return Event(
        event_type=EventType.GOAL_CROSSED,
        evidence=(
            EventEvidence(12, 0.4, (980.0, 300.0)),
            EventEvidence(13, 0.433, (1005.0, 301.0)),
        ),
        confidence=0.82,
        diagnostics=("crossed_right_goal_line",),
    )


def test_event_and_result_serialize_to_native_json_values() -> None:
    """Event reports preserve type, evidence, confidence, and diagnostics."""
    result = EventAnalysisResult(
        config_version="event-analysis-1",
        events=(_event(),),
        warnings=("goal_geometry_requires_review",),
    )

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload == {
        "config_version": "event-analysis-1",
        "events": [
            {
                "event_type": "goal_crossed",
                "evidence": [
                    {
                        "frame_index": 12,
                        "timestamp_seconds": 0.4,
                        "canonical_position": [980.0, 300.0],
                        "observation_state": None,
                        "canonical_player_position": None,
                        "player_observation_state": None,
                    },
                    {
                        "frame_index": 13,
                        "timestamp_seconds": 0.433,
                        "canonical_position": [1005.0, 301.0],
                        "observation_state": None,
                        "canonical_player_position": None,
                        "player_observation_state": None,
                    },
                ],
                "confidence": 0.82,
                "diagnostics": ["crossed_right_goal_line"],
                "details": {},
            }
        ],
        "warnings": ["goal_geometry_requires_review"],
    }


def test_event_evidence_can_retain_a_boundary_without_a_position() -> None:
    """Evidence for an unseen frame retains time and frame without inventing coordinates."""
    evidence = EventEvidence(frame_index=20, timestamp_seconds=0.667)

    assert evidence.to_dict() == {
        "frame_index": 20,
        "timestamp_seconds": 0.667,
        "canonical_position": None,
        "observation_state": None,
        "canonical_player_position": None,
        "player_observation_state": None,
    }


def test_event_evidence_serializes_explicit_ball_observation_state() -> None:
    """Missed and uncertain evidence can be represented without a position."""
    evidence = EventEvidence(
        frame_index=20,
        timestamp_seconds=0.667,
        observation_state=ObservationState.UNCERTAIN,
    )

    assert evidence.to_dict()["observation_state"] == "uncertain"


def test_event_evidence_serializes_measured_player_position() -> None:
    """Proximity evidence can retain ball and player coordinates together."""
    evidence = EventEvidence(
        frame_index=4,
        timestamp_seconds=0.2,
        canonical_position=(501.0, 280.0),
        observation_state=ObservationState.DETECTED,
        canonical_player_position=(500.0, 300.0),
        player_observation_state=PlayerPositionState.DETECTED,
    )

    assert evidence.to_dict()["canonical_player_position"] == [500.0, 300.0]
    assert evidence.to_dict()["player_observation_state"] == "detected"


def test_player_position_track_serializes_as_json_safe_values() -> None:
    """Player measurements retain identity, confidence, and source evidence."""
    observation = PlayerPositionObservation(
        frame_index=2,
        timestamp_seconds=0.1,
        state=PlayerPositionState.DETECTED,
        canonical_position=(500.0, 300.0),
        confidence=0.9,
    )
    track = PlayerPositionTrack("target-player", (observation,), 0.85)

    assert json.loads(json.dumps(track.to_dict())) == track.to_dict()


def test_player_position_track_rejects_out_of_order_observations() -> None:
    """Player inputs must preserve source chronology."""
    earlier = PlayerPositionObservation(0, 0.0, PlayerPositionState.DETECTED, (0.0, 0.0), 0.9)
    later = PlayerPositionObservation(1, 0.1, PlayerPositionState.DETECTED, (0.0, 0.0), 0.9)

    with pytest.raises(ValueError, match="source order"):
        PlayerPositionTrack("target", (later, earlier), 0.9)


def test_event_contracts_are_immutable() -> None:
    """Frozen contracts prevent replacing event and report fields."""
    event = _event()
    result = EventAnalysisResult("event-analysis-1", (event,))

    with pytest.raises(FrozenInstanceError):
        event.confidence = 0.9
    with pytest.raises(FrozenInstanceError):
        result.events = ()
    with pytest.raises(FrozenInstanceError):
        event.evidence[0].frame_index = 100


@pytest.mark.parametrize("frame_index", [-1, True])
def test_event_evidence_rejects_invalid_frame_index(frame_index: int) -> None:
    """Frame indices must be non-negative integers, not booleans."""
    with pytest.raises(ValueError, match="frame_index"):
        EventEvidence(frame_index=frame_index, timestamp_seconds=0.0)


@pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf")])
def test_event_evidence_rejects_invalid_timestamps(timestamp: float) -> None:
    """Evidence timestamps must be finite and non-negative."""
    with pytest.raises(ValueError, match="timestamp_seconds"):
        EventEvidence(frame_index=0, timestamp_seconds=timestamp)


@pytest.mark.parametrize(
    "position",
    [(-1.0, float("nan")), (float("inf"), 2.0), (1.0,)],
)
def test_event_evidence_rejects_invalid_canonical_coordinates(
    position: tuple[float, ...],
) -> None:
    """Canonical positions must be finite two-dimensional points."""
    with pytest.raises(ValueError, match="canonical_position"):
        EventEvidence(frame_index=0, timestamp_seconds=0.0, canonical_position=position)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_event_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    """Event confidence must be a finite value from zero to one."""
    with pytest.raises(ValueError, match="confidence"):
        Event(
            event_type=EventType.BALL_LOST,
            evidence=(EventEvidence(0, 0.0),),
            confidence=confidence,
        )


def test_event_requires_source_evidence() -> None:
    """An event cannot be emitted without at least one attributable frame."""
    with pytest.raises(ValueError, match="evidence"):
        Event(event_type=EventType.BALL_LOST, evidence=(), confidence=0.5)


@pytest.mark.parametrize(
    "evidence",
    [
        (EventEvidence(2, 0.2), EventEvidence(1, 0.3)),
        (EventEvidence(1, 0.3), EventEvidence(2, 0.2)),
    ],
)
def test_event_rejects_out_of_order_evidence(evidence: tuple[EventEvidence, ...]) -> None:
    """Event evidence must be ordered by both source frame and timestamp."""
    with pytest.raises(ValueError, match="ordered"):
        Event(EventType.BALL_REACQUIRED, evidence, 0.5)


def test_result_rejects_events_out_of_chronological_order() -> None:
    """Reports must have a stable chronological event order."""
    later = Event(EventType.BALL_LOST, (EventEvidence(10, 1.0),), 0.5)
    earlier = Event(EventType.BALL_REACQUIRED, (EventEvidence(2, 0.2),), 0.5)

    with pytest.raises(ValueError, match="chronological"):
        EventAnalysisResult("event-analysis-1", (later, earlier))


def test_result_requires_a_configuration_version() -> None:
    """Reports identify the event-analysis configuration that produced them."""
    with pytest.raises(ValueError, match="config_version"):
        EventAnalysisResult("", ())


def test_event_serializes_typed_scalar_details() -> None:
    """Event-specific context such as goal end and direction is JSON-safe."""
    event = Event(
        event_type=EventType.GOAL_CROSSED,
        evidence=(EventEvidence(4, 0.2),),
        confidence=0.9,
        details=(("goal_end", "start"), ("direction", "field_to_goal")),
    )

    assert json.loads(json.dumps(event.to_dict()))["details"] == {
        "goal_end": "start",
        "direction": "field_to_goal",
    }


def test_event_rejects_duplicate_or_non_scalar_details() -> None:
    """Event details cannot serialize ambiguously or leak arbitrary objects."""
    evidence = (EventEvidence(4, 0.2),)
    with pytest.raises(ValueError, match="unique"):
        Event(EventType.GOAL_CROSSED, evidence, 0.9, details=(("end", "start"), ("end", "far")))
    with pytest.raises(ValueError, match="scalar"):
        Event(EventType.GOAL_CROSSED, evidence, 0.9, details=(("coordinates", [1, 2]),))  # type: ignore[arg-type]
