"""Perspective geometry helpers for the canonical foosball table plane."""

from typing import Sequence, Tuple

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import Corners, Point


def field_to_canonical_matrix(
    corners: Corners,
    config: DetectionConfig = DEFAULT_DETECTION_CONFIG,
) -> np.ndarray:
    """Return the perspective matrix mapping image corners to table coordinates."""
    source = np.asarray(corners, dtype=np.float32)
    destination = np.asarray(
        (
            (0.0, 0.0),
            (float(config.canonical_field_width - 1), 0.0),
            (float(config.canonical_field_width - 1), float(config.canonical_field_height - 1)),
            (0.0, float(config.canonical_field_height - 1)),
        ),
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(source, destination)


def canonical_to_field_matrix(
    corners: Corners,
    config: DetectionConfig = DEFAULT_DETECTION_CONFIG,
) -> np.ndarray:
    """Return the perspective matrix mapping table coordinates back to pixels."""
    return np.linalg.inv(field_to_canonical_matrix(corners, config))


def transform_point(point: Point, matrix: np.ndarray) -> Point:
    """Transform one point through a perspective matrix and return finite pixels."""
    source = np.asarray([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(source, matrix)[0, 0]
    return (float(transformed[0]), float(transformed[1]))


def field_to_canonical(point: Point, corners: Corners, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> Point:
    """Map one source-image point into the canonical table plane."""
    return transform_point(point, field_to_canonical_matrix(corners, config))


def canonical_to_field(point: Point, corners: Corners, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> Point:
    """Map one canonical table point back into source-image pixels."""
    return transform_point(point, canonical_to_field_matrix(corners, config))


def order_corners(points: Sequence[Sequence[float]]) -> Corners:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    if len(points) != 4:
        raise ValueError("Exactly four field corners are required")
    array = np.asarray(points, dtype=np.float32)
    y_order = np.argsort(array[:, 1])
    top = array[y_order[:2]]
    bottom = array[y_order[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    ordered = (top[0], top[1], bottom[1], bottom[0])
    return tuple((float(point[0]), float(point[1])) for point in ordered)  # type: ignore[return-value]