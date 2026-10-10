"""Deterministic event extraction from ball-tracking observations."""

from app.ball_tracking.contracts import (
    BallObservation,
    BallTrack,
    ObservationState,
)
from app.detection.contracts.table_contracts import GoalEnd, GoalMouth
from app.event_analysis.contracts import (
    Event,
    EventAnalysisResult,
    EventEvidence,
    EventType,
)
from app.event_analysis.config import DEFAULT_EVENT_ANALYSIS_CONFIG, EventAnalysisConfig


class EventAnalyzer:
    """Convert explicit ball-observation state transitions into events."""

    def __init__(
        self,
        config: EventAnalysisConfig = DEFAULT_EVENT_ANALYSIS_CONFIG,
        config_version: str | None = None,
    ) -> None:
        """Create an analyzer with immutable event thresholds and a version."""
        if config_version is not None and (not isinstance(config_version, str) or not config_version.strip()):
            raise ValueError("config_version must be a non-empty string")
        self._config = config
        self._config_version = config_version or config.config_version

    def analyze(
        self,
        track: BallTrack,
        goal_mouths: tuple[GoalMouth, ...] | None = None,
        goal_warnings: tuple[str, ...] = (),
    ) -> EventAnalysisResult:
        """Emit ball-state and directly supported goal-crossing events.

        A loss begins at the first missed or uncertain observation after a
        detection. Reacquisition is emitted only at a later detection; the full
        unresolved interval remains represented as source evidence. Goal
        crossings require consecutive detected observations and calibrated
        visual aperture geometry.
        """
        self._validate_observation_order(track.observations)
        events: list[Event] = []
        warnings = list(track.warnings)
        warnings.extend(goal_warnings)
        observations = track.observations

        if not observations:
            warnings.append("no_ball_observations")
            warnings.extend(self._goal_geometry_warnings(goal_mouths))
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

        if goal_mouths is None:
            warnings.append("goal_geometry_unavailable")
        else:
            events.extend(self._goal_crossing_events(track, goal_mouths, warnings))

        events.sort(
            key=lambda event: (
                event.evidence[0].timestamp_seconds,
                event.evidence[0].frame_index,
            )
        )
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

    def _goal_crossing_events(
        self,
        track: BallTrack,
        goal_mouths: tuple[GoalMouth, ...],
        warnings: list[str],
    ) -> tuple[Event, ...]:
        """Emit crossings through calibrated mouths, never across unsupported gaps."""
        events: list[Event] = []
        observations = track.observations
        mouths_by_end = {mouth.end: mouth for mouth in goal_mouths}
        for end in GoalEnd:
            mouth = mouths_by_end.get(end)
            if mouth is None:
                warnings.append(f"goal_{end.value}_geometry_unavailable")
                continue
            if mouth.confidence < self._config.minimum_goal_mouth_confidence:
                warnings.append(f"goal_{end.value}_geometry_low_confidence")
                continue

            armed = True
            for previous, current in zip(observations, observations[1:]):
                previous_position = self._canonical_position(previous)
                current_position = self._canonical_position(current)
                if (
                    previous.state is not ObservationState.DETECTED
                    or current.state is not ObservationState.DETECTED
                    or previous_position is None
                    or current_position is None
                ):
                    continue

                if self._is_rearmed(current_position[1], mouth):
                    armed = True
                if current.frame_index != previous.frame_index + 1:
                    continue

                direction = self._crossing_direction(
                    previous_position[1],
                    current_position[1],
                    mouth,
                )
                if direction is None:
                    continue
                crossing_x = self._crossing_x(previous_position, current_position, mouth.crossing_line_y)
                if not mouth.opening_bounds[0] <= crossing_x <= mouth.opening_bounds[2]:
                    continue
                if direction == "field_to_goal" and not armed:
                    continue

                events.append(
                    Event(
                        event_type=EventType.GOAL_CROSSED,
                        evidence=(self._evidence(previous), self._evidence(current)),
                        confidence=min(
                            track.confidence,
                            previous.confidence,
                            current.confidence,
                            mouth.confidence,
                        ),
                        diagnostics=(
                            "estimated_goal_aperture_crossing",
                            f"goal_{end.value}",
                            f"direction_{direction}",
                        ),
                        details=(
                            ("goal_end", end.value),
                            ("direction", direction),
                            ("crossing_x_canonical", crossing_x),
                        ),
                    )
                )
                if direction == "field_to_goal":
                    armed = False
        return tuple(events)

    def _is_rearmed(self, y: float, mouth: GoalMouth) -> bool:
        """Require the ball to return into the field beyond a jitter margin."""
        if mouth.end is GoalEnd.START:
            return y > mouth.crossing_line_y + self._config.goal_rearm_distance_canonical
        return y < mouth.crossing_line_y - self._config.goal_rearm_distance_canonical

    @staticmethod
    def _crossing_direction(previous_y: float, current_y: float, mouth: GoalMouth) -> str | None:
        """Return direction when consecutive points cross the calibrated end line."""
        if mouth.end is GoalEnd.START:
            if previous_y > mouth.crossing_line_y >= current_y:
                return "field_to_goal"
            if previous_y <= mouth.crossing_line_y < current_y:
                return "goal_to_field"
        else:
            if previous_y < mouth.crossing_line_y <= current_y:
                return "field_to_goal"
            if previous_y >= mouth.crossing_line_y > current_y:
                return "goal_to_field"
        return None

    @staticmethod
    def _crossing_x(
        previous_position: tuple[float, float],
        current_position: tuple[float, float],
        crossing_line_y: float,
    ) -> float:
        """Calculate where the measured segment intersects a horizontal goal line."""
        previous_x, previous_y = previous_position
        current_x, current_y = current_position
        fraction = (crossing_line_y - previous_y) / (current_y - previous_y)
        return previous_x + fraction * (current_x - previous_x)

    @staticmethod
    def _canonical_position(observation: BallObservation) -> tuple[float, float] | None:
        """Return only a measured canonical point from a detected observation."""
        if observation.candidate is None:
            return None
        return observation.candidate.canonical_center

    @staticmethod
    def _goal_geometry_warnings(goal_mouths: tuple[GoalMouth, ...] | None) -> tuple[str, ...]:
        """Report missing goal prerequisites when analysis has no ball frames."""
        if goal_mouths is None:
            return ("goal_geometry_unavailable",)
        missing = tuple(
            f"goal_{end.value}_geometry_unavailable"
            for end in GoalEnd
            if not any(mouth.end is end for mouth in goal_mouths)
        )
        return missing

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
