"""Serializable inputs shared by standalone event-analysis components."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from app.ball_tracking.contracts import BallTrack, ObservationState
from app.detection.contracts.table_contracts import TableCalibration

Point = tuple[float, float]


def _finite_float(value: float, field_name: str) -> float:
    """Convert a numeric contract value to a finite native float."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _canonical_point(value: Point, field_name: str) -> Point:
    """Normalize and validate a finite two-dimensional canonical point."""
    try:
        if len(value) != 2:
            raise ValueError
        point = tuple(_finite_float(coordinate, field_name) for coordinate in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a finite 2D point") from error
    return point


class EventType(str, Enum):
    """Supported event categories emitted by event analysis."""

    BALL_LOST = "ball_lost"
    BALL_REACQUIRED = "ball_reacquired"
    GOAL_CROSSED = "goal_crossed"
    BALL_ENTERED_PROXIMITY = "ball_entered_proximity"
    BALL_LEFT_PROXIMITY = "ball_left_proximity"


class PlayerPositionState(str, Enum):
    """Explicit per-frame state for an externally measured player center."""

    DETECTED = "detected"
    MISSED = "missed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class PlayerPositionObservation:
    """One source-frame measurement or explicit unavailable player position."""

    frame_index: int
    timestamp_seconds: float
    state: PlayerPositionState
    canonical_position: Point | None
    confidence: float
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate frame provenance, state, measured center, and confidence."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        timestamp = _finite_float(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        if not isinstance(self.state, PlayerPositionState):
            raise ValueError("state must be a PlayerPositionState")
        if self.canonical_position is None:
            if self.state is PlayerPositionState.DETECTED:
                raise ValueError("canonical_position is required for a detected player")
        else:
            if self.state is not PlayerPositionState.DETECTED:
                raise ValueError("canonical_position is only valid for a detected player")
            object.__setattr__(
                self,
                "canonical_position",
                _canonical_point(self.canonical_position, "canonical_position"),
            )
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if isinstance(self.diagnostics, str):
            raise ValueError("diagnostics must be a sequence of strings")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, str) for item in diagnostics):
            raise ValueError("diagnostics must contain strings")
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return this measured player position as plain JSON-safe values."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "state": self.state.value,
            "canonical_position": (
                list(self.canonical_position) if self.canonical_position is not None else None
            ),
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class PlayerPositionTrack:
    """Ordered measured positions for one stable target-player identity."""

    player_id: str
    observations: tuple[PlayerPositionObservation, ...]
    confidence: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate player identity, ordered observations, confidence, and warnings."""
        if not isinstance(self.player_id, str) or not self.player_id.strip():
            raise ValueError("player_id must be a non-empty string")
        observations = tuple(self.observations)
        if any(not isinstance(item, PlayerPositionObservation) for item in observations):
            raise ValueError("observations must contain PlayerPositionObservation values")
        for previous, current in zip(observations, observations[1:]):
            if (
                current.frame_index <= previous.frame_index
                or current.timestamp_seconds < previous.timestamp_seconds
            ):
                raise ValueError("player observations must be in source order")
        object.__setattr__(self, "observations", observations)
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if isinstance(self.warnings, str):
            raise ValueError("warnings must be a sequence of strings")
        warnings = tuple(self.warnings)
        if any(not isinstance(item, str) for item in warnings):
            raise ValueError("warnings must contain strings")
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return the selected player's ordered position track."""
        return {
            "player_id": self.player_id,
            "observations": [item.to_dict() for item in self.observations],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EventEvidence:
    """One source-frame observation supporting an event."""

    frame_index: int
    timestamp_seconds: float
    canonical_position: Point | None = None
    observation_state: ObservationState | None = None
    canonical_player_position: Point | None = None
    player_observation_state: PlayerPositionState | None = None

    def __post_init__(self) -> None:
        """Validate source evidence and normalize coordinates to Python floats."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        timestamp = _finite_float(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)

        if self.canonical_position is not None:
            object.__setattr__(
                self,
                "canonical_position",
                _canonical_point(self.canonical_position, "canonical_position"),
            )
        if self.observation_state is not None and not isinstance(
            self.observation_state, ObservationState
        ):
            raise ValueError("observation_state must be an ObservationState")
        if self.canonical_player_position is not None:
            object.__setattr__(
                self,
                "canonical_player_position",
                _canonical_point(
                    self.canonical_player_position,
                    "canonical_player_position",
                ),
            )
        if self.player_observation_state is not None and not isinstance(
            self.player_observation_state,
            PlayerPositionState,
        ):
            raise ValueError("player_observation_state must be a PlayerPositionState")

    def to_dict(self) -> dict[str, Any]:
        """Return frame, timestamp, and optional coordinates as native JSON values."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "canonical_position": (
                list(self.canonical_position) if self.canonical_position is not None else None
            ),
            "observation_state": (
                self.observation_state.value if self.observation_state is not None else None
            ),
            "canonical_player_position": (
                list(self.canonical_player_position)
                if self.canonical_player_position is not None
                else None
            ),
            "player_observation_state": (
                self.player_observation_state.value
                if self.player_observation_state is not None
                else None
            ),
        }


@dataclass(frozen=True)
class Event:
    """An event with ordered source evidence, confidence, and diagnostics."""

    event_type: EventType
    evidence: tuple[EventEvidence, ...]
    confidence: float
    diagnostics: tuple[str, ...] = ()
    details: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def __post_init__(self) -> None:
        """Validate event evidence, chronological order, and confidence."""
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(item, EventEvidence) for item in evidence):
            raise ValueError("evidence must contain at least one EventEvidence item")
        for previous, current in zip(evidence, evidence[1:]):
            if (
                current.frame_index <= previous.frame_index
                or current.timestamp_seconds < previous.timestamp_seconds
            ):
                raise ValueError("event evidence must be ordered by frame and timestamp")
        object.__setattr__(self, "evidence", evidence)

        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        object.__setattr__(self, "confidence", confidence)

        if isinstance(self.diagnostics, str):
            raise ValueError("diagnostics must be a sequence of strings")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(diagnostic, str) for diagnostic in diagnostics):
            raise ValueError("diagnostics must contain strings")
        object.__setattr__(self, "diagnostics", diagnostics)
        details = tuple(self.details)
        detail_names: set[str] = set()
        for detail in details:
            if not isinstance(detail, tuple) or len(detail) != 2:
                raise ValueError("details must contain key/value pairs")
            name, value = detail
            if not isinstance(name, str) or not name or name in detail_names:
                raise ValueError("detail names must be non-empty and unique")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("event details must contain JSON scalar values")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("event detail numbers must be finite")
            detail_names.add(name)
        object.__setattr__(self, "details", details)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe event report."""
        return {
            "event_type": self.event_type.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class EventAnalysisResult:
    """Events and diagnostics produced under one event-analysis configuration."""

    config_version: str
    events: tuple[Event, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the result version and preserve events in chronological order."""
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ValueError("config_version must be a non-empty string")
        events = tuple(self.events)
        if any(not isinstance(event, Event) for event in events):
            raise ValueError("events must contain Event values")
        event_keys = tuple(
            (event.evidence[0].timestamp_seconds, event.evidence[0].frame_index)
            for event in events
        )
        if event_keys != tuple(sorted(event_keys)):
            raise ValueError("events must be in chronological order")
        object.__setattr__(self, "events", events)

        warnings = tuple(self.warnings)
        if isinstance(self.warnings, str) or any(not isinstance(warning, str) for warning in warnings):
            raise ValueError("warnings must contain strings")
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete event-analysis report as JSON-safe values."""
        return {
            "config_version": self.config_version,
            "events": [event.to_dict() for event in self.events],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EventAnalysisInput:
    """Bundle ball, calibration, and optional measured player-track inputs.

    Ball and calibration values remain their existing upstream contracts.
    Player positions are optional measured input; this model does not detect
    or infer them.
    """

    track: BallTrack
    calibration: TableCalibration
    player_track: PlayerPositionTrack | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe ball, calibration, and optional player inputs."""
        return {
            "track": self.track.to_dict(),
            "calibration": self.calibration.to_dict(),
            "player_track": (
                self.player_track.to_dict() if self.player_track is not None else None
            ),
        }
