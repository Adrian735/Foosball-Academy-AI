"""Render standalone ball-tracking diagnostics onto video frames."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.ball_tracking.contracts import BallCandidate, BallObservation, ObservationState
from app.detection.contracts.table_contracts import FieldGeometry

_FIELD_COLOUR = (0, 255, 255)
_ACCEPTED_COLOUR = (255, 180, 0)
_REJECTED_COLOUR = (0, 0, 255)
_DETECTED_COLOUR = (0, 220, 0)
_MISSED_COLOUR = (0, 220, 220)
_UNCERTAIN_COLOUR = (0, 80, 255)
_TEXT_COLOUR = (255, 255, 255)


def render_ball_tracking_frame(
    image: np.ndarray,
    field_geometry: FieldGeometry,
    observation: BallObservation,
    candidates: Sequence[BallCandidate] = (),
    previous_observation: BallObservation | None = None,
) -> np.ndarray:
    """Return a copy annotated with one frame's tracking diagnostics."""
    overlay = image.copy()
    _draw_field(overlay, field_geometry)
    for candidate in candidates:
        colour = _ACCEPTED_COLOUR if not candidate.diagnostics else _REJECTED_COLOUR
        _draw_candidate(overlay, candidate, colour, 1)

    if (
        previous_observation is not None
        and previous_observation.state is ObservationState.DETECTED
        and observation.state is ObservationState.DETECTED
        and previous_observation.candidate is not None
        and observation.candidate is not None
    ):
        cv2.line(
            overlay,
            _point(previous_observation.candidate),
            _point(observation.candidate),
            _DETECTED_COLOUR,
            2,
            cv2.LINE_AA,
        )

    if observation.candidate is not None:
        colour = {
            ObservationState.DETECTED: _DETECTED_COLOUR,
            ObservationState.MISSED: _MISSED_COLOUR,
            ObservationState.UNCERTAIN: _UNCERTAIN_COLOUR,
        }[observation.state]
        cv2.circle(overlay, _point(observation.candidate), 7, colour, -1, cv2.LINE_AA)
    _draw_status(overlay, observation)
    return overlay


def write_debug_image(output_path: str | Path, image: np.ndarray) -> None:
    """Write one BGR debug image and fail explicitly if encoding fails."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise ValueError(f"Unable to write debug image: {destination}")


def _draw_field(image: np.ndarray, field_geometry: FieldGeometry) -> None:
    """Draw the calibrated field polygon."""
    points = np.asarray(field_geometry.corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, _FIELD_COLOUR, 2, cv2.LINE_AA)


def _draw_candidate(
    image: np.ndarray,
    candidate: BallCandidate,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a pixel-space candidate centre and bounding box."""
    x, y, width, height = candidate.bounding_box
    cv2.rectangle(
        image,
        (int(x), int(y)),
        (int(x + width), int(y + height)),
        colour,
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(image, _point(candidate), 3, colour, -1, cv2.LINE_AA)


def _draw_status(image: np.ndarray, observation: BallObservation) -> None:
    """Draw source position, state, confidence, and diagnostics."""
    status = (
        f"frame: {observation.frame_index} | "
        f"time: {observation.timestamp_seconds:.3f}s | "
        f"state: {observation.state.value} | "
        f"confidence: {observation.confidence:.2f}"
    )
    cv2.putText(image, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOUR, 2, cv2.LINE_AA)
    for index, diagnostic in enumerate(observation.diagnostics):
        cv2.putText(
            image,
            diagnostic,
            (12, 54 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _UNCERTAIN_COLOUR,
            1,
            cv2.LINE_AA,
        )


def _point(candidate: BallCandidate) -> tuple[int, int]:
    """Convert a candidate's pixel centre into an OpenCV point."""
    return int(round(candidate.center[0])), int(round(candidate.center[1]))