"""Render table-calibration diagnostics onto BGR images."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.detection.contracts.rod_contracts import Rod, RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry, GoalMouth, TableCalibration
from app.detection.geometry import canonical_to_field

_FIELD_COLOUR = (0, 255, 255)
_ROD_COLOUR = (255, 0, 255)
_GOAL_COLOUR = (255, 255, 0)
_GOAL_LINE_COLOUR = (0, 165, 255)
_REJECTED_ROD_COLOUR = (0, 0, 255)
_TEXT_COLOUR = (255, 255, 255)
_WARNING_COLOUR = (0, 0, 255)
_STATUS_X_POSITION = 12
_CONFIDENCE_Y_POSITION = 28
_WARNING_START_Y_POSITION = 56
_WARNING_LINE_HEIGHT = 24
_CONFIDENCE_FONT_SCALE = 0.7
_WARNING_FONT_SCALE = 0.55


def render_table_calibration(image: np.ndarray, calibration: TableCalibration) -> np.ndarray:
    """Return a BGR copy of ``image`` annotated with calibration results."""
    overlay = image.copy()
    if calibration.field:
        _draw_field(overlay, calibration.field.to_dict()["corners"])
    _draw_rods(overlay, [rod.to_dict() for rod in calibration.rods])
    if calibration.field is not None:
        _draw_goal_mouths(overlay, calibration.field, calibration.goal_mouths)
    _draw_status(overlay, calibration.confidence, calibration.warnings)
    return overlay


def render_goal_mouths(
    image: np.ndarray,
    field: FieldGeometry,
    goal_mouths: Sequence[GoalMouth],
) -> np.ndarray:
    """Return a copy annotated with calibrated aperture bounds and crossing lines."""
    overlay = image.copy()
    _draw_goal_mouths(overlay, field, goal_mouths)
    return overlay


def render_expected_annotation(image: np.ndarray, expected: Mapping[str, Any]) -> np.ndarray:
    """Return a BGR copy of ``image`` annotated from an expected-fixture mapping."""
    overlay = image.copy()
    corners = expected.get("field_corners", [])
    if len(corners) == 4:
        _draw_field(overlay, corners)
    rod_lines = expected.get("rod_lines", [])
    if rod_lines:
        _draw_rods(overlay, [{"index": index, "line": line} for index, line in enumerate(rod_lines)])
    else:
        _draw_rod_positions(overlay, expected.get("rod_y_positions", []), corners)
    return overlay


def render_field_detection(
    image: np.ndarray,
    field: FieldGeometry | None,
    frame_index: int,
    diagnostics: Sequence[str] = (),
) -> np.ndarray:
    """Return an image annotated with one automated field-detection result."""
    overlay = image.copy()
    if field is not None:
        _draw_field(overlay, field.to_dict()["corners"])
        status = (
            f"frame: {frame_index} | method: {field.detection_method} | "
            f"confidence: {field.confidence:.3f}"
        )
        _draw_status(overlay, field.confidence, (status, *diagnostics))
    else:
        _draw_status(overlay, 0.0, (f"frame: {frame_index} | field: not detected", *diagnostics))
    return overlay


def render_rod_candidates(
    image: np.ndarray,
    accepted_candidates: Sequence[RodCandidate],
    rejected_candidates: Sequence[RodCandidate] = (),
) -> np.ndarray:
    """Return a copy annotated with accepted and rejected per-frame rod lines."""
    overlay = image.copy()
    for candidate in rejected_candidates:
        _draw_candidate_line(overlay, candidate, _REJECTED_ROD_COLOUR, 1)
    for candidate in accepted_candidates:
        _draw_candidate_line(overlay, candidate, _ROD_COLOUR, 2)
    return overlay


def render_stable_rods(image: np.ndarray, rods: Sequence[Rod]) -> np.ndarray:
    """Return a copy annotated with rods recovered across sampled frames."""
    overlay = image.copy()
    _draw_rods(overlay, [rod.to_dict() for rod in rods])
    return overlay


def write_debug_image(output_path: str, image: np.ndarray) -> None:
    """Write a BGR debug image, raising ``ValueError`` if encoding fails."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise ValueError(f"Unable to write debug image: {destination}")


def _draw_field(image: np.ndarray, corners: Sequence[Sequence[float]]) -> None:
    """Draw an ordered four-corner field polygon and corner indices."""
    points = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, _FIELD_COLOUR, 2, cv2.LINE_AA)
    for index, point in enumerate(points.reshape((-1, 2))):
        position = (int(point[0]), int(point[1]))
        cv2.circle(image, position, 5, _FIELD_COLOUR, -1, cv2.LINE_AA)
        cv2.putText(image, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FIELD_COLOUR, 2, cv2.LINE_AA)


def _draw_rods(image: np.ndarray, rods: Sequence[Mapping[str, Any]]) -> None:
    """Draw rod segments and their stable indices."""
    for rod in rods:
        start, end = rod["line"]
        start_point = (int(start[0]), int(start[1]))
        end_point = (int(end[0]), int(end[1]))
        cv2.line(image, start_point, end_point, _ROD_COLOUR, 2, cv2.LINE_AA)
        midpoint = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        cv2.putText(
            image,
            str(rod["index"]),
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _ROD_COLOUR,
            2,
            cv2.LINE_AA,
        )


def _draw_goal_mouths(
    image: np.ndarray,
    field: FieldGeometry,
    goal_mouths: Sequence[GoalMouth],
) -> None:
    """Draw each canonical aperture box and its crossing plane in source pixels."""
    for mouth in goal_mouths:
        left, top, right, bottom = mouth.opening_bounds
        canonical_points = (
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        )
        points = np.asarray(
            [
                canonical_to_field(point, field.corners)
                for point in canonical_points
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, _GOAL_COLOUR, 2, cv2.LINE_AA)
        line_start = canonical_to_field((left, mouth.crossing_line_y), field.corners)
        line_end = canonical_to_field((right, mouth.crossing_line_y), field.corners)
        cv2.line(
            image,
            (int(round(line_start[0])), int(round(line_start[1]))),
            (int(round(line_end[0])), int(round(line_end[1]))),
            _GOAL_LINE_COLOUR,
            3,
            cv2.LINE_AA,
        )
        line_y = int(round((line_start[1] + line_end[1]) / 2.0))
        label_y = line_y + 18 if line_y < image.shape[0] // 2 else line_y - 8
        cv2.putText(
            image,
            f"goal line {mouth.end.value}",
            (
                max(2, int(round(min(line_start[0], line_end[0])))),
                min(image.shape[0] - 2, max(14, label_y)),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _GOAL_LINE_COLOUR,
            2,
            cv2.LINE_AA,
        )


def _draw_candidate_line(
    image: np.ndarray,
    candidate: RodCandidate,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw one detector candidate without exposing its internal line type."""
    start, end = candidate.line
    cv2.line(
        image,
        (int(start[0]), int(start[1])),
        (int(end[0]), int(end[1])),
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _draw_rod_positions(image: np.ndarray, positions: Sequence[float], corners: Sequence[Sequence[float]]) -> None:
    """Draw expected rod positions across the field bounds when lines are absent."""
    if len(corners) == 4:
        x_positions = [point[0] for point in corners]
        start_x, end_x = int(min(x_positions)), int(max(x_positions))
    else:
        start_x, end_x = 0, image.shape[1] - 1
    rods = [
        {"index": index, "line": [[start_x, position], [end_x, position]]}
        for index, position in enumerate(positions)
    ]
    _draw_rods(image, rods)


def _draw_status(image: np.ndarray, confidence: float, warnings: Sequence[str]) -> None:
    """Draw calibration confidence and warnings in the image margin."""
    cv2.putText(
        image,
        f"confidence: {confidence:.2f}",
        (_STATUS_X_POSITION, _CONFIDENCE_Y_POSITION),
        cv2.FONT_HERSHEY_SIMPLEX,
        _CONFIDENCE_FONT_SCALE,
        _TEXT_COLOUR,
        2,
        cv2.LINE_AA,
    )
    for index, warning in enumerate(warnings):
        cv2.putText(
            image,
            warning,
            (_STATUS_X_POSITION, _WARNING_START_Y_POSITION + index * _WARNING_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            _WARNING_FONT_SCALE,
            _WARNING_COLOUR,
            1,
            cv2.LINE_AA,
        )