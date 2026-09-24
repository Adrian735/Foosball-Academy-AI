"""Temporal association for stateless ball detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

from app.ball_tracking.config import BallTrackingConfig, DEFAULT_BALL_TRACKING_CONFIG
from app.ball_tracking.contracts import (
    BallCandidate,
    BallObservation,
    BallTrack,
    BallTrackingResult,
    ObservationState,
)
from app.ball_tracking.detector import BallDetectionFrame, BallDetector
from app.ball_tracking.frame_reader import SequentialFrame
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.contracts.video_contracts import VideoMetadata


@dataclass(frozen=True)
class _TrackState:
    """Last accepted measurement used for temporal association."""

    candidate: BallCandidate
    timestamp_seconds: float


class BallTracker:
    """Associate frame-local ball candidates without predicting measurements."""

    def __init__(
        self,
        detector: Optional[BallDetector] = None,
        config: BallTrackingConfig = DEFAULT_BALL_TRACKING_CONFIG,
    ) -> None:
        """Create a tracker with an immutable association configuration."""
        self._config = config
        self._detector = detector or BallDetector(config)

    def track(
        self,
        frames: Iterable[SequentialFrame],
        field_geometry: FieldGeometry,
        video_metadata: VideoMetadata | dict[str, object],
        calibration_config_version: str,
        decoder_warnings: tuple[str, ...] = (),
    ) -> BallTrackingResult:
        """Detect and associate every supplied frame in source order.

        Missing frames remain explicit observations. A candidate outside the
        time-scaled displacement bound becomes uncertain and cannot move the
        accepted trajectory.
        """
        observations: list[BallObservation] = []
        candidates_by_frame: list[tuple[BallCandidate, ...]] = []
        last_state: Optional[_TrackState] = None
        missing_start: Optional[float] = None
        longest_missing_interval = 0.0
        detected_confidences: list[float] = []

        for frame in frames:
            detection = self._detector.detect(frame.image, field_geometry)
            candidates_by_frame.append(detection.accepted + detection.rejected)
            observation, last_state, missing_start, longest_missing_interval = self._associate(
                frame,
                detection,
                last_state,
                missing_start,
                longest_missing_interval,
            )
            observations.append(observation)
            if observation.state is ObservationState.DETECTED:
                detected_confidences.append(observation.confidence)

        if observations and missing_start is not None:
            longest_missing_interval = max(
                longest_missing_interval,
                observations[-1].timestamp_seconds - missing_start,
            )

        observation_count = len(observations)
        detected_count = sum(
            observation.state is ObservationState.DETECTED for observation in observations
        )
        coverage = detected_count / observation_count if observation_count else 0.0
        mean_confidence = (
            sum(detected_confidences) / len(detected_confidences)
            if detected_confidences
            else 0.0
        )
        confidence = mean_confidence * coverage
        warnings = list(decoder_warnings)
        if coverage < self._config.minimum_track_coverage:
            warnings.append("track_coverage_below_threshold")
        if longest_missing_interval > self._config.maximum_unresolved_gap_seconds:
            warnings.append("unresolved_missing_interval")
        metadata = (
            video_metadata.to_dict()
            if isinstance(video_metadata, VideoMetadata)
            else dict(video_metadata)
        )
        track = BallTrack(
            video_metadata=metadata,
            calibration_config_version=calibration_config_version,
            tracking_config_version=self._config.config_version,
            observations=tuple(observations),
            detection_coverage=coverage,
            longest_missing_interval_seconds=longest_missing_interval,
            confidence=confidence,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return BallTrackingResult(track, tuple(candidates_by_frame))

    def _associate(
        self,
        frame: SequentialFrame,
        detection: BallDetectionFrame,
        last_state: Optional[_TrackState],
        missing_start: Optional[float],
        longest_missing_interval: float,
    ) -> tuple[BallObservation, Optional[_TrackState], Optional[float], float]:
        """Select one candidate or preserve the frame as missing/uncertain."""
        if last_state is None:
            if not detection.accepted:
                return self._missing_observation(
                    frame, "no_accepted_candidate", None, missing_start, longest_missing_interval
                )
            candidate = detection.accepted[0]
            return (
                BallObservation(
                    frame.frame_index,
                    frame.timestamp_seconds,
                    ObservationState.DETECTED,
                    candidate,
                    candidate.confidence,
                ),
                _TrackState(candidate, frame.timestamp_seconds),
                None,
                longest_missing_interval,
            )

        elapsed = max(0.0, frame.timestamp_seconds - last_state.timestamp_seconds)
        maximum_distance = self._config.maximum_displacement_canonical_per_second * elapsed
        if not detection.accepted:
            return self._missing_observation(
                frame, "no_accepted_candidate", last_state, missing_start, longest_missing_interval
            )
        if (
            missing_start is not None
            and elapsed > self._config.maximum_recovery_gap_seconds
        ):
            return (
                BallObservation(
                    frame.frame_index,
                    frame.timestamp_seconds,
                    ObservationState.UNCERTAIN,
                    None,
                    0.0,
                    ("recovery_gap_exceeded",),
                ),
                last_state,
                missing_start,
                longest_missing_interval,
            )
        selected = self._select_candidate(detection.accepted, last_state.candidate, maximum_distance)
        if selected is not None:
            if missing_start is not None:
                longest_missing_interval = max(
                    longest_missing_interval,
                    frame.timestamp_seconds - missing_start,
                )
            association_confidence = self._association_confidence(
                selected,
                last_state.candidate,
                maximum_distance,
            )
            return (
                BallObservation(
                    frame.frame_index,
                    frame.timestamp_seconds,
                    ObservationState.DETECTED,
                    selected,
                    association_confidence,
                ),
                _TrackState(selected, frame.timestamp_seconds),
                None,
                longest_missing_interval,
            )

        return (
            BallObservation(
                frame.frame_index,
                frame.timestamp_seconds,
                ObservationState.UNCERTAIN,
                None,
                0.0,
                ("implausible_jump",),
            ),
            last_state,
            missing_start,
            longest_missing_interval,
        )

    def _select_candidate(
        self,
        candidates: tuple[BallCandidate, ...],
        previous: BallCandidate,
        maximum_distance: float,
    ) -> Optional[BallCandidate]:
        """Choose the best candidate within the elapsed-time movement bound."""
        if maximum_distance <= 0:
            return min(candidates, key=lambda candidate: self._distance(candidate, previous), default=None)
        eligible = [
            candidate
            for candidate in candidates
            if self._distance(candidate, previous) <= maximum_distance + 1e-9
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda candidate: self._association_confidence(candidate, previous, maximum_distance),
        )

    def _association_confidence(
        self,
        candidate: BallCandidate,
        previous: BallCandidate,
        maximum_distance: float,
    ) -> float:
        """Combine intrinsic candidate quality with continuity quality."""
        distance = self._distance(candidate, previous)
        continuity = 1.0 if maximum_distance <= 0 else max(0.0, 1.0 - distance / maximum_distance)
        total_weight = self._config.circularity_score_weight + self._config.continuity_score_weight
        if total_weight <= 0:
            return candidate.confidence
        return (
            self._config.circularity_score_weight * candidate.confidence
            + self._config.continuity_score_weight * continuity
        ) / total_weight

    @staticmethod
    def _distance(first: BallCandidate, second: BallCandidate) -> float:
        """Return canonical-plane distance between two candidates."""
        if first.canonical_center is None or second.canonical_center is None:
            return float("inf")
        return hypot(
            first.canonical_center[0] - second.canonical_center[0],
            first.canonical_center[1] - second.canonical_center[1],
        )

    def _missing_observation(
        self,
        frame: SequentialFrame,
        diagnostic: str,
        state: Optional[_TrackState],
        missing_start: Optional[float],
        longest_missing_interval: float,
    ) -> tuple[BallObservation, Optional[_TrackState], Optional[float], float]:
        """Create a missing observation and start its unresolved interval."""
        if missing_start is None:
            missing_start = frame.timestamp_seconds
        return (
            BallObservation(
                frame.frame_index,
                frame.timestamp_seconds,
                ObservationState.MISSED,
                None,
                0.0,
                (diagnostic,),
            ),
            state,
            missing_start,
            longest_missing_interval,
        )