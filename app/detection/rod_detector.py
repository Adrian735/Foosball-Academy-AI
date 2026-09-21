"""Detect per-frame horizontal rod candidates inside a calibrated field."""

from math import hypot
from typing import Sequence

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.rod_contracts import RodCandidate, RodDetectionResult
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
        return list(self.detect_frame(frame, field_geometry).accepted)

    def detect_frame(self, frame: np.ndarray, field_geometry: FieldGeometry) -> RodDetectionResult:
        """Return accepted and rejected candidates for one calibrated frame."""
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
            return RodDetectionResult(())

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
            if not self._is_near_horizontal(candidate):
                continue
            if self._is_accepted(candidate):
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        merged_rejected = self._merge_fragments(frame, field_geometry, rejected, hsv, gray)
        promoted = [candidate for candidate in merged_rejected if self._is_accepted(candidate)]
        self._last_rejected_candidates = tuple(candidate for candidate in merged_rejected if not self._is_accepted(candidate))
        return RodDetectionResult(
            accepted=tuple(self._deduplicate_candidates([*accepted, *promoted])),
            rejected=self._last_rejected_candidates,
        )

    def _score_line(
        self,
        frame: np.ndarray,
        field_geometry: FieldGeometry,
        raw_line: tuple[float, float, float, float],
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
        shadow_score = self._line_shadow_score(frame, start, end, gray_image)
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
            f"shadow_band={shadow_score:.3f}",
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

    @staticmethod
    def _is_near_horizontal(candidate: RodCandidate) -> bool:
        """Exclude non-rod Hough lines before same-row fragment clustering."""
        angle_diagnostic = next(item for item in candidate.diagnostics if item.startswith("angle_score="))
        return float(angle_diagnostic.removeprefix("angle_score=")) > 0.0

    def _is_accepted(self, candidate: RodCandidate) -> bool:
        """Require both a useful score and enough span to represent a physical rod."""
        outside_field = candidate.field_relative_y < 0.0 or candidate.field_relative_y > 1.0
        shadow_score = float(next(item for item in candidate.diagnostics if item.startswith("shadow_band=")).removeprefix("shadow_band="))
        if (
            not outside_field
            and shadow_score >= self._config.minimum_shadow_band_score
            and candidate.player_colour_evidence < self._config.minimum_shadow_colour_evidence
        ):
            return False
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

    @staticmethod
    def _line_shadow_score(
        frame: np.ndarray,
        start: Point,
        end: Point,
        gray_image: np.ndarray | None = None,
    ) -> float:
        """Measure whether a candidate is a dark band relative to parallel neighbours."""
        gray = gray_image if gray_image is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        direction = np.asarray((end[0] - start[0], end[1] - start[1]), dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return 0.0
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float32) / length
        offset = normal * 9.0

        def band_mean(displacement: np.ndarray) -> float:
            first = np.asarray(start, dtype=np.float32) + displacement
            last = np.asarray(end, dtype=np.float32) + displacement
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.line(
                mask,
                (int(round(first[0])), int(round(first[1]))),
                (int(round(last[0])), int(round(last[1]))),
                255,
                7,
            )
            values = gray[mask > 0]
            return float(np.mean(values)) if values.size else 0.0

        centre = band_mean(np.zeros(2, dtype=np.float32))
        neighbours = max(band_mean(offset), band_mean(-offset))
        return float(np.clip((neighbours - centre) / 255.0, 0.0, 1.0))

    def _merge_fragments(
        self,
        frame: np.ndarray,
        field_geometry: FieldGeometry,
        candidates: list[RodCandidate],
        hsv_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> list[RodCandidate]:
        """Fit and rescore one physical line from same-row Hough fragments."""
        merged: list[list[RodCandidate]] = []
        separation = self._config.minimum_rod_cluster_separation
        regions = (
            tuple(candidate for candidate in candidates if candidate.field_relative_y < 0.0),
            tuple(candidate for candidate in candidates if 0.0 <= candidate.field_relative_y <= 1.0),
            tuple(candidate for candidate in candidates if candidate.field_relative_y > 1.0),
        )
        for region in regions:
            region_clusters: list[list[RodCandidate]] = []
            for candidate in sorted(region, key=lambda item: item.field_relative_y):
                cluster = next((cluster for cluster in region_clusters if abs(cluster[-1].field_relative_y - candidate.field_relative_y) <= separation), None)
                if cluster is None:
                    region_clusters.append([candidate])
                else:
                    cluster.append(candidate)
            merged.extend(region_clusters)
        result: list[RodCandidate] = []
        for cluster in merged:
            line = self._merged_line(cluster)
            candidate = self._score_line(
                frame,
                field_geometry,
                line,
                hsv_image=hsv_image,
                gray_image=gray_image,
            )
            if candidate is not None:
                result.append(candidate)
        return result

    def _deduplicate_candidates(self, candidates: list[RodCandidate]) -> list[RodCandidate]:
        """Keep the strongest candidate for each physical rod row."""
        merged: list[list[RodCandidate]] = []
        separation = self._config.minimum_rod_cluster_separation
        regions = (
            tuple(candidate for candidate in candidates if candidate.field_relative_y < 0.0),
            tuple(candidate for candidate in candidates if 0.0 <= candidate.field_relative_y <= 1.0),
            tuple(candidate for candidate in candidates if candidate.field_relative_y > 1.0),
        )
        for region in regions:
            region_clusters: list[list[RodCandidate]] = []
            for candidate in sorted(region, key=lambda item: item.field_relative_y):
                cluster = next((cluster for cluster in region_clusters if abs(cluster[-1].field_relative_y - candidate.field_relative_y) <= separation), None)
                if cluster is None:
                    region_clusters.append([candidate])
                else:
                    cluster.append(candidate)
            merged.extend(region_clusters)
        return [max(cluster, key=lambda item: item.confidence) for cluster in merged]

    @staticmethod
    def _merged_line(candidates: Sequence[RodCandidate]) -> tuple[float, float, float, float]:
        """Fit a single segment spanning the endpoints of a Hough fragment cluster."""
        points = np.asarray([point for candidate in candidates for point in candidate.line], dtype=np.float32)
        if len(points) == 2:
            return (*points[0], *points[1])
        direction_x, direction_y, origin_x, origin_y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        direction = np.asarray((direction_x, direction_y), dtype=np.float32)
        origin = np.asarray((origin_x, origin_y), dtype=np.float32)
        projections = (points - origin) @ direction
        start = origin + direction * float(np.min(projections))
        end = origin + direction * float(np.max(projections))
        return (float(start[0]), float(start[1]), float(end[0]), float(end[1]))