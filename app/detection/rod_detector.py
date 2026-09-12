"""Detect per-frame horizontal rod candidates inside a calibrated field."""

from math import hypot

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.rod_contracts import RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry, Point
from app.detection.geometry import field_to_canonical


class RodDetector:
    """Find and score physical rod line candidates without deciding rod identity."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a detector using immutable line, geometry, and colour thresholds."""
        self._config = config
        self._last_rejected_candidates: tuple[RodCandidate, ...] = ()

    @property
    def rejected_candidates(self) -> tuple[RodCandidate, ...]:
        """Return rejected candidates from the most recent frame for diagnostics."""
        return self._last_rejected_candidates

    def detect(self, frame: np.ndarray, field_geometry: FieldGeometry) -> list[RodCandidate]:
        """Return scored rod candidates whose lines satisfy the configured geometry gates."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Rod detection expects a BGR color image")

        field_points = np.asarray(field_geometry.corners, dtype=np.int32)
        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(search_mask, field_points, 255)
        margin = self._config.rod_search_margin_pixels
        search_mask = cv2.dilate(search_mask, np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._config.canny_low_threshold, self._config.canny_high_threshold)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        edges = cv2.bitwise_and(edges, search_mask)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self._config.hough_threshold,
            minLineLength=self._config.hough_min_line_length,
            maxLineGap=self._config.hough_max_line_gap,
        )
        if lines is None:
            self._last_rejected_candidates = ()
            return []

        accepted: list[RodCandidate] = []
        rejected: list[RodCandidate] = []
        for raw_line in lines.reshape(-1, 4):
            candidate = self._score_line(
                frame,
                field_geometry,
                tuple(int(value) for value in raw_line),
                hsv_image=hsv,
                gray_image=gray,
            )
            if candidate is None:
                continue
            if self._is_accepted(candidate):
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        self._last_rejected_candidates = tuple(rejected)
        return self._merge_fragments(accepted)

    def _score_line(
        self,
        frame: np.ndarray,
        field_geometry: FieldGeometry,
        raw_line: tuple[int, int, int, int],
        hsv_image: np.ndarray | None = None,
        gray_image: np.ndarray | None = None,
    ) -> RodCandidate | None:
        """Convert one Hough segment into a JSON-safe scored candidate."""
        start = (float(raw_line[0]), float(raw_line[1]))
        end = (float(raw_line[2]), float(raw_line[3]))
        canonical_start = field_to_canonical(start, field_geometry.corners, self._config)
        canonical_end = field_to_canonical(end, field_geometry.corners, self._config)
        canonical_length = hypot(canonical_end[0] - canonical_start[0], canonical_end[1] - canonical_start[1])
        if canonical_length <= 0:
            return None
        angle = abs(float(np.degrees(np.arctan2(
            canonical_end[1] - canonical_start[1],
            canonical_end[0] - canonical_start[0],
        ))))
        if angle > 90:
            angle = 180 - angle
        angle_score = max(0.0, 1.0 - angle / self._config.maximum_rod_angle_degrees)
        length_ratio = min(1.0, canonical_length / self._config.canonical_field_width)
        overlap_score = self._field_overlap_score(field_geometry.corners, start, end)
        colour_evidence = self._player_colour_evidence(frame, start, end, hsv_image)
        brightness_score = self._line_brightness_score(frame, start, end, gray_image)
        confidence = (
            0.30 * angle_score
            + 0.30 * length_ratio
            + 0.20 * overlap_score
            + 0.10 * colour_evidence
            + 0.10 * brightness_score
        )
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        canonical_midpoint = field_to_canonical(midpoint, field_geometry.corners, self._config)
        relative_y = canonical_midpoint[1] / max(1.0, float(self._config.canonical_field_height - 1))
        diagnostics = (
            f"angle_score={angle_score:.3f}",
            f"length_ratio={length_ratio:.3f}",
            f"field_overlap={overlap_score:.3f}",
            f"player_colour={colour_evidence:.3f}",
            f"line_quality={brightness_score:.3f}",
        )
        return RodCandidate(
            line=(start, end),
            field_relative_y=float(relative_y),
            length_ratio=float(length_ratio),
            player_colour_evidence=float(colour_evidence),
            confidence=float(confidence),
            diagnostics=diagnostics,
        )

    def _minimum_candidate_confidence(self) -> float:
        """Return the minimum score required for a useful rod line."""
        return 0.45

    def _is_accepted(self, candidate: RodCandidate) -> bool:
        """Require both a useful score and enough span to represent a physical rod."""
        outside_field = candidate.field_relative_y < 0.0 or candidate.field_relative_y > 1.0
        return (
            candidate.confidence >= (
                self._config.minimum_goal_rod_confidence
                if outside_field
                else self._minimum_candidate_confidence()
            )
            and candidate.length_ratio >= (
                self._config.minimum_goal_rod_length_ratio
                if outside_field
                else self._config.minimum_rod_length_ratio
            )
        )

    @staticmethod
    def _field_overlap_score(corners: tuple[Point, ...], start: Point, end: Point) -> float:
        """Estimate the fraction of a line segment inside the playable field."""
        polygon = np.asarray(corners, dtype=np.float32)
        samples = np.linspace(0.0, 1.0, 25)
        inside = 0
        for ratio in samples:
            point = (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
            inside += int(cv2.pointPolygonTest(polygon, point, False) >= 0)
        return float(inside / len(samples))

    def _player_colour_evidence(
        self,
        frame: np.ndarray,
        start: Point,
        end: Point,
        hsv_image: np.ndarray | None = None,
    ) -> float:
        """Measure red/blue pixels in a narrow band around a candidate line."""
        hsv = hsv_image if hsv_image is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_lower, red_upper = self._config.red_hsv_range
        red_wrap_lower, red_wrap_upper = self._config.red_wrap_hsv_range
        blue_lower, blue_upper = self._config.blue_hsv_range
        colour_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.asarray(red_lower), np.asarray(red_upper)),
            cv2.inRange(hsv, np.asarray(red_wrap_lower), np.asarray(red_wrap_upper)),
        )
        colour_mask = cv2.bitwise_or(colour_mask, cv2.inRange(hsv, np.asarray(blue_lower), np.asarray(blue_upper)))
        band = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(band, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, 15)
        band_size = np.count_nonzero(band)
        return float(np.count_nonzero(cv2.bitwise_and(colour_mask, band)) / band_size) if band_size else 0.0

    @staticmethod
    def _line_brightness_score(
        frame: np.ndarray,
        start: Point,
        end: Point,
        gray_image: np.ndarray | None = None,
    ) -> float:
        """Score visible line quality from grayscale contrast in a narrow band."""
        gray = gray_image if gray_image is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(band, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, 5)
        values = gray[band > 0]
        if values.size == 0:
            return 0.0
        return float(np.clip(np.std(values) / 64.0, 0.0, 1.0))

    def _merge_fragments(self, candidates: list[RodCandidate]) -> list[RodCandidate]:
        """Merge nearby Hough fragments while preserving the strongest diagnostics."""
        merged: list[list[RodCandidate]] = []
        separation = self._config.minimum_rod_cluster_separation
        for candidate in sorted(candidates, key=lambda item: item.field_relative_y):
            cluster = next((cluster for cluster in merged if abs(cluster[-1].field_relative_y - candidate.field_relative_y) <= separation), None)
            if cluster is None:
                merged.append([candidate])
            else:
                cluster.append(candidate)
        result: list[RodCandidate] = []
        for cluster in merged:
            strongest = max(cluster, key=lambda item: item.confidence)
            result.append(strongest)
        return result