"""Manual field-detection diagnostics for all supplied table videos.

Run with ``pytest -s tests/manual/test_field_detection_manual.py`` so the
per-video debug output remains visible for human inspection.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from app.detection.field_detector import FieldDetector
from app.detection.debug_renderer import render_field_detection, write_debug_image

VIDEO_DIRECTORY = Path("tests/table-detection_tests")
EXPECTED_DIRECTORY = Path("tests/fixtures/expected")
STARTUP_FRAME_LIMIT = 90
SAMPLE_COUNT = 5
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "tests/artifacts/field_detection"
DEBUG_OUTPUT_DIRECTORY = Path(
    os.environ.get("FIELD_DEBUG_OUTPUT_DIR", str(DEFAULT_DEBUG_OUTPUT_DIRECTORY))
)
MANUAL_FRAME_LABELS = {
    "table-detection_test-1": {
        0: ("wrong", ("top_right", "bottom_left")),
        22: ("wrong", ("top_right",)),
        44: ("good", ()),
        66: ("wrong", ("top_left",)),
        89: ("good", ()),
    },
    "table-detection_test-2": {
        0: ("good", ()),
        22: ("wrong", ("top_right", "top_left")),
        44: ("good", ()),
        66: ("wrong", ("bottom_left",)),
        89: ("wrong", ("top_right",)),
    },
    "table-detection_test-3": {
        0: ("unusable", ()),
        22: ("unusable", ()),
        44: ("good", ()),
        66: ("good", ()),
        89: ("unusable", ()),
    },
    "table-detection_test-4": {
        0: ("wrong", ("top_right", "top_left")),
        22: ("good", ()),
        44: ("wrong", ("top_right", "top_left")),
        66: ("wrong", ("top_right", "top_left")),
        89: ("wrong", ("top_right", "top_left")),
    },
}


def _expected_path(video_path: Path) -> Path:
    """Find the annotation JSON associated with one supplied video."""
    direct_path = EXPECTED_DIRECTORY / f"{video_path.stem}.json"
    if direct_path.exists():
        return direct_path
    frame_annotation = EXPECTED_DIRECTORY / f"{video_path.stem}-frame-30.json"
    if frame_annotation.exists():
        return frame_annotation
    raise FileNotFoundError(f"No expected annotation found for {video_path}")


def _sample_indices(frame_count: int) -> tuple[int, ...]:
    """Return evenly spaced frame indices from the startup window."""
    window_size = min(frame_count, STARTUP_FRAME_LIMIT)
    return tuple(int(index) for index in np.linspace(0, window_size - 1, SAMPLE_COUNT, dtype=int))


def _corner_errors(actual: tuple[tuple[float, float], ...], expected: list[list[int]]) -> list[float]:
    """Calculate per-corner Euclidean errors against manual annotations."""
    return [
        ((detected[0] - target[0]) ** 2 + (detected[1] - target[1]) ** 2) ** 0.5
        for detected, target in zip(actual, expected)
    ]


def test_field_detection_manual_diagnostics_for_all_videos() -> None:
    """Print and save field-detection diagnostics for every supplied video."""
    detector = FieldDetector()
    video_paths = sorted(VIDEO_DIRECTORY.glob("*.mp4"))
    assert len(video_paths) == 4, "Expected exactly four supplied table-detection videos"

    for video_path in video_paths:
        expected = json.loads(_expected_path(video_path).read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(str(video_path))
        assert capture.isOpened(), f"Unable to open {video_path}"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = _sample_indices(frame_count)
        detections = []
        video_output_directory = DEBUG_OUTPUT_DIRECTORY / video_path.stem
        video_output_directory.mkdir(parents=True, exist_ok=True)
        labels = MANUAL_FRAME_LABELS[video_path.stem]
        assert set(labels) == set(frame_indices), f"Manual labels do not cover {video_path.name}"
        print(f"\n[{video_path.name}] frame_count={frame_count} sample_indices={frame_indices}")
        try:
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                was_read, frame = capture.read()
                assert was_read and frame is not None, f"Unable to read frame {frame_index}"
                detection = detector.detect(frame)
                manual_status, wrong_corners = labels[frame_index]
                manual_note = f"manual={manual_status}"
                if wrong_corners:
                    manual_note += f" wrong_corners={list(wrong_corners)}"
                if detection is None:
                    print(f"  frame={frame_index} detected=None {manual_note}")
                    debug_image = render_field_detection(frame, None, frame_index, (manual_note,))
                    debug_path = video_output_directory / f"frame-{frame_index:04d}.png"
                    write_debug_image(str(debug_path), debug_image)
                    print(f"    debug_image={debug_path}")
                    continue
                detections.append(detection)
                errors = _corner_errors(detection.corners, expected["field_corners"])
                debug_image = render_field_detection(
                    frame,
                    detection,
                    frame_index,
                    (
                        manual_note,
                        f"corner errors: {[round(error, 1) for error in errors]}",
                    ),
                )
                debug_path = video_output_directory / f"frame-{frame_index:04d}.png"
                write_debug_image(str(debug_path), debug_image)
                print(
                    f"  frame={frame_index} method={detection.detection_method} "
                    f"confidence={detection.confidence:.3f} corners={detection.corners} "
                    f"corner_errors={[round(error, 1) for error in errors]} {manual_note} "
                    f"debug_image={debug_path}"
                )
        finally:
            capture.release()

        assert detections, f"No field geometry detected in {video_path.name}"
    summary = {status: sum(label[0] == status for label in labels.values()) for status in ("good", "wrong", "unusable")}
    print(f"  manual_summary={summary}")