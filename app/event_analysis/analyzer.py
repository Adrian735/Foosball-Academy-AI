"""Deterministic event extraction from ball-tracking observations."""

from app.ball_tracking.contracts import (
    BallObservation,
    BallTrack,
    ObservationState,
)
from app.event_analysis.contracts import (
    Event,
    EventAnalysisResult,
    EventEvidence,
    EventType,
)


class EventAnalyzer:
    """Convert explicit ball-observation state transitions into events."""

    def __init__(self, config_version: str = "event-analysis-1") -> None:
        """Create an analyzer tagged with its behavior/configuration version."""
        if not isinstance(config_version, str) or not config_version.strip():
            raise ValueError("config_version must be a non-empty string")
        self._config_version = config_version

    def analyze(self, track: BallTrack) -> EventAnalysisResult:
        """Emit lost/reacquired events without interpolating unseen positions.

        A loss begins at the first missed or uncertain observation after a
        detection. Reacquisition is emitted only at a later detection; the full
        unresolved interval remains represented as source evidence.
        """
        self._validate_observation_order(track.observations)
        events: list[Event] = []
        warnings = list(track.warnings)
        observations = track.observations

        if not observations:
            warnings.append("no_ball_observations")
            return EventAnalysisResult(
                self._config_version,
                tuple(events),
                self._unique(warnings),
            )

        if observations[0].state is not ObservationState.DETECTED:
            warnings.append("ball_unobserved_at_start")

        last_detected: BallObservation | None = None
        gap_observations: list[BallObservation] = []

        for observation in observations:
            if observation.state is ObservationState.DETECTED:
                if gap_observations and last_detected is not None:
                    events.append(
                        Event(
                            event_type=EventType.BALL_REACQUIRED,
                            evidence=tuple(
                                self._evidence(item)
                                for item in (
                                    last_detected,
                                    *gap_observations,
                                    observation,
                                )
                            ),
                            confidence=min(
                                track.confidence,
                                last_detected.confidence,
                                observation.confidence,
                            ),
                            diagnostics=self._reacquisition_diagnostics(gap_observations),
                        )
                    )
                    gap_observations.clear()
                last_detected = observation
                continue

            if last_detected is None:
                continue

            if not gap_observations:
                events.append(
                    Event(
                        event_type=EventType.BALL_LOST,
                        evidence=(
                            self._evidence(last_detected),
                            self._evidence(observation),
                        ),
                        confidence=min(
                            track.confidence,
                            last_detected.confidence,
                            observation.confidence,
                        ),
                        diagnostics=self._loss_diagnostics(observation),
                    )
                )
            gap_observations.append(observation)

        if gap_observations:
            warnings.append("unresolved_ball_loss_interval")

        return EventAnalysisResult(
            config_version=self._config_version,
            events=tuple(events),
            warnings=self._unique(warnings),
        )

    @staticmethod
    def _validate_observation_order(observations: tuple[BallObservation, ...]) -> None:
        """Reject tracks whose source frame or timestamp order is inconsistent."""
        for previous, current in zip(observations, observations[1:]):
            if (
                current.frame_index <= previous.frame_index
                or current.timestamp_seconds < previous.timestamp_seconds
            ):
                raise ValueError("ball observations must be in source order")

    @staticmethod
    def _evidence(observation: BallObservation) -> EventEvidence:
        """Copy source metadata without inventing a position for unseen frames."""
        candidate = observation.candidate
        position = candidate.canonical_center if candidate is not None else None
        return EventEvidence(
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            canonical_position=position,
            observation_state=observation.state,
        )

    @staticmethod
    def _loss_diagnostics(observation: BallObservation) -> tuple[str, ...]:
        """Explain why the ball-loss boundary was emitted."""
        return EventAnalyzer._unique(
            ("ball_observation_not_detected", *observation.diagnostics)
        )

    @staticmethod
    def _reacquisition_diagnostics(
        gap_observations: list[BallObservation],
    ) -> tuple[str, ...]:
        """Retain tracking diagnostics accumulated over the unseen interval."""
        diagnostics = ["ball_reacquired_after_unobserved_interval"]
        for observation in gap_observations:
            diagnostics.extend(observation.diagnostics)
        return EventAnalyzer._unique(diagnostics)

    @staticmethod
    def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Return strings once while preserving their original order."""
        return tuple(dict.fromkeys(values))
