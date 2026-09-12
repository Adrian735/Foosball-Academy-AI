"""Detect the playable field polygon in one calibration frame."""

from typing import Optional

import cv2
import numpy as np

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.geometry import order_corners


class FieldDetector:
    """Detect a Bonzini playable-field quadrilateral from a BGR frame."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a detector using immutable HSV and morphology thresholds."""
        self._config = config

    def detect(self, frame: np.ndarray) -> Optional[FieldGeometry]:
        """Return one field candidate, or ``None`` when no valid contour exists."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Field detection expects a BGR color image")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._field_mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        frame_area = float(frame.shape[0] * frame.shape[1])
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        if area_ratio < self._config.minimum_field_area_ratio:
            return None

        selection = self._select_corners(mask, contour, frame.shape[1], frame.shape[0])
        if selection is None:
            return None

        corners, method, contour_iou = selection

        ordered = order_corners(corners)
        polygon_area = abs(float(cv2.contourArea(np.asarray(ordered, dtype=np.float32))))
        if polygon_area <= 0:
            return None
        x, y, width, height = cv2.boundingRect(np.asarray(ordered, dtype=np.float32))
        rectangularity = min(1.0, polygon_area / max(1.0, float(width * height)))
        area_score = min(1.0, area_ratio / 0.5)
        confidence = min(
            1.0,
            self._config.field_area_confidence_weight * area_score
            + self._config.field_rectangularity_confidence_weight * rectangularity
            + self._config.field_contour_agreement_confidence_weight * contour_iou,
        )
        return FieldGeometry(ordered, (x, y, width, height), confidence, method)

    def _field_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create and clean the combined green/cyan HSV field mask."""
        green_lower, green_upper = self._config.green_hsv_range
        cyan_lower, cyan_upper = self._config.cyan_hsv_range
        green_mask = cv2.inRange(hsv, np.asarray(green_lower), np.asarray(green_upper))
        cyan_mask = cv2.inRange(hsv, np.asarray(cyan_lower), np.asarray(cyan_upper))
        mask = cv2.bitwise_or(green_mask, cyan_mask)
        kernel_size = self._config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _select_corners(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Optional[tuple[tuple[tuple[float, float], ...], str, float]]:
        """Select the candidate quadrilateral that best agrees with the field contour."""
        hough_corners = self._hough_corners(mask, contour, frame_width, frame_height)
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        candidates: list[tuple[tuple[tuple[float, float], ...], str]] = []
        if hough_corners is not None:
            candidates.append((hough_corners, "hough_intersections"))
        if len(approximation) >= 4:
            points = approximation.reshape(-1, 2)
            candidates.append((self._extreme_points(points), "contour_approximation"))

        rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
        if len(rectangle) == 4:
            candidates.append((tuple((float(point[0]), float(point[1])) for point in rectangle), "min_area_rectangle"))

        x, y, width, height = cv2.boundingRect(contour)
        candidates.append((
            ((float(x), float(y)), (float(x + width), float(y)),
             (float(x + width), float(y + height)), (float(x), float(y + height))),
            "bounding_rectangle",
        ))

        scored = [
            (corners, method, self._contour_iou(mask, corners))
            for corners, method in candidates
        ]
        valid = [
            candidate
            for candidate in scored
            if candidate[1] != "hough_intersections"
            or candidate[2] >= self._config.minimum_hough_contour_iou
        ]
        return max(valid, key=lambda candidate: candidate[2], default=None)

    @staticmethod
    def _contour_iou(mask: np.ndarray, corners: tuple[tuple[float, float], ...]) -> float:
        """Return the IoU between a candidate quadrilateral and the cleaned field mask."""
        candidate_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(candidate_mask, np.asarray(corners, dtype=np.int32), 255)
        intersection = np.count_nonzero(cv2.bitwise_and(mask, candidate_mask))
        union = np.count_nonzero(cv2.bitwise_or(mask, candidate_mask))
        return float(intersection / union) if union else 0.0

    def _hough_corners(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Optional[tuple[tuple[float, float], ...]]:
        """Find four boundary intersections from strong horizontal/vertical edges."""
        x, y, width, height = cv2.boundingRect(contour)
        margin = 20
        x_start, y_start = max(0, x - margin), max(0, y - margin)
        x_end = min(frame_width, x + width + margin)
        y_end = min(frame_height, y + height + margin)
        edges = cv2.Canny(mask[y_start:y_end, x_start:x_end], self._config.canny_low_threshold, self._config.canny_high_threshold)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self._config.hough_threshold,
            minLineLength=self._config.hough_min_line_length,
            maxLineGap=self._config.hough_max_line_gap,
        )
        if lines is None:
            return None
        horizontal: list[np.ndarray] = []
        vertical: list[np.ndarray] = []
        for line in lines.reshape(-1, 4):
            angle = abs(float(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))))
            if angle < 30 or angle > 150:
                horizontal.append(line)
            elif 60 < angle < 120:
                vertical.append(line)
        if len(horizontal) < 2 or len(vertical) < 2:
            return None
        top = min(horizontal, key=lambda line: float(line[1] + line[3]))
        bottom = max(horizontal, key=lambda line: float(line[1] + line[3]))
        left = min(vertical, key=lambda line: float(line[0] + line[2]))
        right = max(vertical, key=lambda line: float(line[0] + line[2]))
        intersections = (
            self._line_intersection(top, left), self._line_intersection(top, right),
            self._line_intersection(bottom, right), self._line_intersection(bottom, left),
        )
        if any(point is None for point in intersections):
            return None
        return tuple((point[0] + x_start, point[1] + y_start) for point in intersections if point is not None)

    @staticmethod
    def _line_intersection(first: np.ndarray, second: np.ndarray) -> Optional[tuple[float, float]]:
        """Return the intersection of two infinite line segments."""
        x1, y1, x2, y2 = (float(value) for value in first)
        x3, y3, x4, y4 = (float(value) for value in second)
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-8:
            return None
        factor = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        return (x1 + factor * (x2 - x1), y1 + factor * (y2 - y1))

    @staticmethod
    def _extreme_points(points: np.ndarray) -> tuple[tuple[float, float], ...]:
        """Reduce an approximated contour to four extreme corner points."""
        return (
            tuple(float(value) for value in points[np.argmin(points[:, 0] + points[:, 1])]),
            tuple(float(value) for value in points[np.argmax(points[:, 0] - points[:, 1])]),
            tuple(float(value) for value in points[np.argmax(points[:, 0] + points[:, 1])]),
            tuple(float(value) for value in points[np.argmin(points[:, 0] - points[:, 1])]),
        )