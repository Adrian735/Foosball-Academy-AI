"""Focused tests for the standalone runner and debug renderer."""

import json

import numpy as np

from app.ball_tracking.contracts import BallCandidate, BallObservation, ObservationState
from app.ball_tracking.debug_renderer import render_ball_tracking_frame
from app.ball_tracking.runner import main
from app.detection.contracts.table_contracts import FieldGeometry


FIELD = FieldGeometry(
    corners=((2.0, 2.0), (97.0, 2.0), (97.0, 97.0), (2.0, 97.0)),
    bounding_box=(2, 2, 95, 95),
    confidence=1.0,
    detection_method="synthetic",
)


def _candidate(x: float) -> BallCandidate:
    """Build a small synthetic pixel-space candidate."""
    return BallCandidate((x, 50.0), (x, 50.0), (x - 2, 48, 4, 4), 10.0, 0.9, 1.0, 0.9)


def _observation(index: int, state: ObservationState, candidate: BallCandidate | None) -> BallObservation:
    """Build an observation for renderer tests."""
    return BallObservation(index, index / 10.0, state, candidate, 0.9 if candidate else 0.0)


def test_renderer_draws_field_and_contiguous_trajectory() -> None:
    """Adjacent detections draw a visible trajectory between their centres."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = render_ball_tracking_frame(
        image,
        FIELD,
        _observation(1, ObservationState.DETECTED, _candidate(30.0)),
        (_candidate(30.0),),
        _observation(0, ObservationState.DETECTED, _candidate(10.0)),
    )

    assert rendered is not image
    assert tuple(rendered[50, 20]) == (0, 220, 0)
    assert not np.array_equal(rendered, image)


def test_renderer_does_not_bridge_missing_observation() -> None:
    """A missed frame prevents a trajectory segment from spanning the gap."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = render_ball_tracking_frame(
        image,
        FIELD,
        _observation(1, ObservationState.MISSED, None),
        (),
        _observation(0, ObservationState.DETECTED, _candidate(10.0)),
    )

    assert tuple(rendered[50, 20]) == (0, 0, 0)


def test_runner_reports_invalid_calibration_as_json(tmp_path, capsys) -> None:
    """Malformed calibration exits non-zero with a machine-readable error."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            "--video",
            "missing.mp4",
            "--calibration",
            str(calibration_path),
            "--output",
            str(tmp_path / "track.json"),
        ]
    )

    assert exit_code == 2
    assert json.loads(capsys.readouterr().err)["error"].startswith("Invalid calibration")