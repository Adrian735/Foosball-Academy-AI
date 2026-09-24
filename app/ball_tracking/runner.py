"""Command-line entry point for standalone ball tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.ball_tracking.calibration import CalibrationLoadError, load_table_calibration, require_trackable_calibration
from app.ball_tracking.contracts import BallTrackingResult
from app.ball_tracking.debug_renderer import render_ball_tracking_frame, write_debug_image
from app.ball_tracking.frame_reader import BallVideoReadError, SequentialFrame, SequentialFrameReader
from app.ball_tracking.tracker import BallTracker
from app.detection.contracts.table_contracts import FieldGeometry


def main(argv: list[str] | None = None) -> int:
    """Run ball tracking and write a JSON report, returning a process status."""
    arguments = _parse_arguments(argv)
    try:
        calibration = require_trackable_calibration(load_table_calibration(arguments.calibration))
        read_result = SequentialFrameReader().read(arguments.video)
        if calibration.field is None:
            raise CalibrationLoadError("Calibration has no field geometry")
        tracking_result = BallTracker().track(
            read_result.frames,
            calibration.field,
            read_result.metadata,
            calibration.detector_config_version,
            read_result.warnings,
        )
        _write_json(arguments.output, tracking_result.to_dict())
        if arguments.debug_output_dir:
            _write_debug_frames(arguments.debug_output_dir, read_result.frames, calibration.field, tracking_result)
        return 0
    except (CalibrationLoadError, BallVideoReadError, OSError, ValueError) as error:
        _write_error(str(error))
        return 2


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the standalone runner command-line arguments."""
    parser = argparse.ArgumentParser(description="Track a yellow ball on a calibrated foosball table")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--calibration", required=True, help="Validated table calibration JSON path")
    parser.add_argument("--output", required=True, help="Output ball-track JSON path")
    parser.add_argument("--debug-output-dir", help="Optional directory for per-frame debug PNGs")
    return parser.parse_args(argv)


def _write_json(output_path: str, payload: dict[str, object]) -> None:
    """Write one JSON-safe report and create its parent directory."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_error(message: str) -> None:
    """Write an actionable JSON-safe error report to standard error."""
    json.dump({"error": message}, sys.stderr)
    sys.stderr.write("\n")


def _write_debug_frames(
    output_dir: str,
    frames: tuple[SequentialFrame, ...],
    field_geometry: FieldGeometry,
    tracking_result: BallTrackingResult,
) -> None:
    """Render one deterministic PNG per decoded source frame."""
    output_path = Path(output_dir)
    observations = tracking_result.track.observations
    candidates_by_frame = tracking_result.candidates_by_frame
    for index, frame in enumerate(frames):
        previous = observations[index - 1] if index else None
        rendered = render_ball_tracking_frame(
            frame.image,
            field_geometry,
            observations[index],
            candidates_by_frame[index],
            previous,
        )
        write_debug_image(output_path / f"frame-{frame.frame_index:06d}.png", rendered)


if __name__ == "__main__":
    raise SystemExit(main())