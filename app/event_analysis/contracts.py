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


class EventType(str, Enum):
    """Supported event categories emitted by event analysis."""

    BALL_LOST = "ball_lost"
    BALL_REACQUIRED = "ball_reacquired"
    GOAL_CROSSED = "goal_crossed"
    BALL_ENTERED_PROXIMITY = "ball_entered_proximity"
    BALL_LEFT_PROXIMITY = "ball_left_proximity"


@dataclass(frozen=True)
class EventEvidence:
    """One source-frame observation supporting an event."""

    frame_index: int
    timestamp_seconds: float
    canonical_position: Point | None = None
    observation_state: ObservationState | None = None

    def __post_init__(self) -> None:
        """Validate source evidence and normalize coordinates to Python floats."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        timestamp = _finite_float(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)

        if self.canonical_position is not None:
            try:
                if len(self.canonical_position) != 2:
                    raise ValueError
                position = tuple(float(coordinate) for coordinate in self.canonical_position)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("canonical_position must be a finite 2D point") from error
            if not all(isfinite(coordinate) for coordinate in position):
                raise ValueError("canonical_position must be a finite 2D point")
            object.__setattr__(self, "canonical_position", position)
        if self.observation_state is not None and not isinstance(
            self.observation_state, ObservationState
        ):
            raise ValueError("observation_state must be an ObservationState")

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
        }


@dataclass(frozen=True)
class Event:
    """An event with ordered source evidence, confidence, and diagnostics."""

    event_type: EventType
    evidence: tuple[EventEvidence, ...]
    confidence: float
    diagnostics: tuple[str, ...] = ()

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe event report."""
        return {
            "event_type": self.event_type.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
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
