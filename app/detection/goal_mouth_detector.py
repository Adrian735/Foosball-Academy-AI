"""Visually detect goal apertures and combine their geometry across frames."""

from collections.abc import Sequence
from statistics import median

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import FieldGeometry, GoalEnd, GoalMouth
from app.detection.geometry import field_to_canonical_matrix


class GoalMouthDetector:
    """Detect dark goal apertures within an extended canonical table warp."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a detector and consensus combiner using versioned thresholds."""
        self._config = config

    def detect_frame(
        self,
        frame: np.ndarray,
        field_geometry: FieldGeometry,
    ) -> tuple[GoalMouth, ...]:
        """Detect zero, one, or two goal apertures from one calibrated frame."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Goal-mouth detection expects a BGR color image")
        if self._config.goal_aperture_search_depth_canonical <= 0:
            raise ValueError("Goal-aperture search depth must be positive")

        width = self._config.canonical_field_width
        height = self._config.canonical_field_height
        depth = self._config.goal_aperture_search_depth_canonical
        translate = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, float(depth)), (0.0, 0.0, 1.0)))
        transform = translate @ field_to_canonical_matrix(field_geometry.corners, self._config)
        warped = cv2.warpPerspective(frame, transform, (width, height + 2 * depth))
        value = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = cv2.inRange(value, 0, self._config.goal_aperture_max_value)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        detections = []
        for end in GoalEnd:
            candidate = self._detect_end(dark, end, width, height, depth)
            if candidate is not None:
                detections.append(candidate)
        return tuple(
            sorted(
                detections,
                key=lambda mouth: 0 if mouth.end is GoalEnd.START else 1,
            )
        )

    def combine(
        self,
        candidates_by_frame: Sequence[Sequence[GoalMouth]],
    ) -> tuple[tuple[GoalMouth, ...], tuple[str, ...]]:
        """Return stable per-end bounds and explicit diagnostics for unavailable ends."""
        mouths: list[GoalMouth] = []
        warnings: list[str] = []
        for end in GoalEnd:
            candidates = [
                mouth
                for frame_mouths in candidates_by_frame
                for mouth in frame_mouths
                if mouth.end is end
            ]
            if len(candidates) < self._config.goal_aperture_minimum_consensus_frames:
                warnings.append(f"goal_{end.value}_aperture_not_confidently_visible")
                continue

            median_bounds = tuple(
                float(median(mouth.opening_bounds[index] for mouth in candidates))
                for index in range(4)
            )
            inliers = [
                mouth
                for mouth in candidates
                if all(
                    abs(mouth.opening_bounds[index] - median_bounds[index])
                    <= self._config.goal_aperture_maximum_consensus_spread
                    for index in range(4)
                )
            ]
            if len(inliers) < self._config.goal_aperture_minimum_consensus_frames:
                warnings.append(f"goal_{end.value}_aperture_unstable")
                continue
            bounds = tuple(
                float(median(mouth.opening_bounds[index] for mouth in inliers))
                for index in range(4)
            )

            confidence = float(median(mouth.confidence for mouth in inliers))
            if confidence < self._config.minimum_goal_aperture_confidence:
                warnings.append(f"goal_{end.value}_aperture_low_confidence")
                continue
            crossing_line_y = 0.0 if end is GoalEnd.START else float(
                self._config.canonical_field_height - 1
            )
            mouths.append(
                GoalMouth(
                    end=end,
                    opening_bounds=bounds,
                    crossing_line_y=crossing_line_y,
                    confidence=confidence,
                    diagnostics=("visual_aperture_consensus",),
                )
            )
        return tuple(mouths), tuple(warnings)

    def _detect_end(
        self,
        dark_mask: np.ndarray,
        end: GoalEnd,
        width: int,
        height: int,
        depth: int,
    ) -> GoalMouth | None:
        """Select the best plausible centered dark component at one field end."""
        field_edge = depth if end is GoalEnd.START else depth + height - 1
        if end is GoalEnd.START:
            y_start, y_end = 0, min(dark_mask.shape[0], depth * 2 + 1)
        else:
            y_start, y_end = max(0, field_edge - depth), min(dark_mask.shape[0], field_edge + depth + 1)
        components = cv2.connectedComponentsWithStats(dark_mask[y_start:y_end], 8)
        count, _, stats, _ = components
        best: tuple[float, GoalMouth] | None = None
        for label in range(1, count):
            x, relative_y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if area < self._config.goal_aperture_min_area_canonical:
                continue
            width_ratio = component_width / width
            if not self._config.goal_aperture_min_width_ratio <= width_ratio <= self._config.goal_aperture_max_width_ratio:
                continue
            center_x = x + component_width / 2.0
            center_error = abs(center_x / width - 0.5)
            center_tolerance = self._config.goal_aperture_center_tolerance_ratio
            if center_error > center_tolerance or component_height <= 0:
                continue
            component_y = y_start + relative_y - depth
            component_y_end = component_y + component_height
            boundary = 0.0 if end is GoalEnd.START else float(height - 1)
            boundary_distance = max(component_y - boundary, boundary - component_y_end, 0.0)
            boundary_tolerance = self._config.goal_aperture_boundary_tolerance_canonical
            if boundary_distance > boundary_tolerance:
                continue
            if component_width / component_height < 1.2:
                continue

            width_midpoint = (
                self._config.goal_aperture_min_width_ratio
                + self._config.goal_aperture_max_width_ratio
            ) / 2.0
            width_half_range = (
                self._config.goal_aperture_max_width_ratio
                - self._config.goal_aperture_min_width_ratio
            ) / 2.0
            width_score = max(0.0, 1.0 - abs(width_ratio - width_midpoint) / width_half_range)
            center_score = max(0.0, 1.0 - center_error / center_tolerance)
            boundary_score = max(0.0, 1.0 - boundary_distance / boundary_tolerance)
            area_score = min(1.0, area / (self._config.goal_aperture_min_area_canonical * 4.0))
            confidence = (width_score + center_score + boundary_score + area_score) / 4.0
            mouth = GoalMouth(
                end=end,
                opening_bounds=(
                    float(x),
                    float(component_y),
                    float(x + component_width),
                    float(component_y_end),
                ),
                crossing_line_y=boundary,
                confidence=confidence,
                diagnostics=("dark_centered_aperture_candidate",),
            )
            if best is None or confidence > best[0]:
                best = (confidence, mouth)
        return best[1] if best is not None else None
