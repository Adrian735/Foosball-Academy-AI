"""Consensus and stability checks for per-frame field candidates."""

from typing import Iterable

import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.geometry import order_corners


class FieldConsensusError(ValueError):
    """Raised when sampled field candidates cannot form stable consensus."""


class FieldConsensus:
    """Combine stable per-frame field detections into one geometry candidate."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a consensus combiner using configured confidence and spread limits."""
        self._config = config

    def combine(self, candidates: Iterable[FieldGeometry]) -> FieldGeometry:
        """Return median corners or raise when candidates are insufficient/unstable."""
        usable = [candidate for candidate in candidates if candidate.confidence >= self._config.minimum_field_confidence]
        if len(usable) < self._config.minimum_accepted_frames:
            raise FieldConsensusError("Too few confident field candidates for consensus")
        corner_array = np.asarray([candidate.corners for candidate in usable], dtype=np.float32)
        median_corners = np.median(corner_array, axis=0)
        candidate_distances = np.max(np.linalg.norm(corner_array - median_corners, axis=2), axis=1)
        inlier_mask = candidate_distances <= self._config.maximum_field_corner_spread_pixels
        inlier_array = corner_array[inlier_mask]
        if len(inlier_array) < self._config.minimum_accepted_frames:
            raise FieldConsensusError("Field corner spread exceeds the configured stability tolerance")
        median_corners = np.median(inlier_array, axis=0)
        spread = np.max(np.linalg.norm(inlier_array - median_corners, axis=2), axis=0)
        if float(np.max(spread)) > self._config.maximum_field_corner_spread_pixels:
            raise FieldConsensusError("Field corner spread exceeds the configured stability tolerance")
        ordered = order_corners(median_corners.tolist())
        x_values = [point[0] for point in ordered]
        y_values = [point[1] for point in ordered]
        bounding_box = (
            int(round(min(x_values))),
            int(round(min(y_values))),
            int(round(max(x_values) - min(x_values))),
            int(round(max(y_values) - min(y_values))),
        )
        confidence = float(np.median([candidate.confidence for candidate in usable]))
        return FieldGeometry(ordered, bounding_box, confidence, "frame_consensus")