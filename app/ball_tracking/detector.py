"""Stateless, field-constrained yellow-ball candidate detection."""

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np

from app.ball_tracking.config import BallTrackingConfig, DEFAULT_BALL_TRACKING_CONFIG
from app.ball_tracking.contracts import BallCandidate
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.geometry import field_to_canonical


@dataclass(frozen=True)
class BallDetectionFrame:
    """Accepted and rejected ball candidates from one source frame."""

    accepted: Tuple[BallCandidate, ...]
    rejected: Tuple[BallCandidate, ...]
    mask: Any = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe candidate diagnostics without the image mask."""
        return {
            "accepted": [candidate.to_dict() for candidate in self.accepted],
            "rejected": [candidate.to_dict() for candidate in self.rejected],
        }


class BallDetector:
    """Detect yellow ball-shaped contours without temporal state."""

    def __init__(self, config: BallTrackingConfig = DEFAULT_BALL_TRACKING_CONFIG) -> None:
        """Create a detector using immutable ball-tracking thresholds."""
        self._config = config

    def detect(self, frame: np.ndarray, field_geometry: FieldGeometry) -> BallDetectionFrame:
        """Return field-constrained yellow candidates for one BGR frame."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Ball detection requires a BGR colour image")

        field_mask = self._field_mask(frame.shape[:2], field_geometry)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = self._config.yellow_hsv_range
        yellow_mask = cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8))
        mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=field_mask)
        kernel_size = self._config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        accepted: list[BallCandidate] = []
        rejected: list[BallCandidate] = []
        frame_area = float(frame.shape[0] * frame.shape[1])
        for contour in contours:
            candidate, reasons = self._candidate(contour, field_geometry, frame_area)
            if reasons:
                rejected.append(
                    BallCandidate(
                        center=candidate.center,
                        canonical_center=candidate.canonical_center,
                        bounding_box=candidate.bounding_box,
                        area=candidate.area,
                        circularity=candidate.circularity,
                        aspect_ratio=candidate.aspect_ratio,
                        confidence=candidate.confidence,
                        diagnostics=tuple(reasons),
                    )
                )
            elif candidate.confidence >= self._config.minimum_detection_confidence:
                accepted.append(candidate)
            else:
                rejected.append(
                    BallCandidate(
                        center=candidate.center,
                        canonical_center=candidate.canonical_center,
                        bounding_box=candidate.bounding_box,
                        area=candidate.area,
                        circularity=candidate.circularity,
                        aspect_ratio=candidate.aspect_ratio,
                        confidence=candidate.confidence,
                        diagnostics=("below_confidence_gate",),
                    )
                )
        accepted.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return BallDetectionFrame(tuple(accepted), tuple(rejected), mask)

    def _candidate(
        self,
        contour: np.ndarray,
        field_geometry: FieldGeometry,
        frame_area: float,
    ) -> tuple[BallCandidate, list[str]]:
        """Build one candidate and named rejection reasons from a contour."""
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        center = (float(x + width / 2.0), float(y + height / 2.0))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
        aspect_ratio = float(width / height) if height else float("inf")
        canonical_center = field_to_canonical(center, field_geometry.corners)
        area_ratio = area / frame_area
        aspect_score = min(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 0.0
        confidence = float(0.7 * min(circularity, 1.0) + 0.3 * aspect_score)
        candidate = BallCandidate(
            center=center,
            canonical_center=canonical_center,
            bounding_box=(float(x), float(y), float(width), float(height)),
            area=area,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            confidence=confidence,
        )
        reasons: list[str] = []
        if area_ratio < self._config.minimum_contour_area_ratio:
            reasons.append("contour_too_small")
        if area_ratio > self._config.maximum_contour_area_ratio:
            reasons.append("contour_too_large")
        if circularity < self._config.minimum_circularity:
            reasons.append("insufficient_circularity")
        if aspect_ratio < self._config.minimum_aspect_ratio or aspect_ratio > self._config.maximum_aspect_ratio:
            reasons.append("invalid_aspect_ratio")
        canonical_x, canonical_y = canonical_center
        maximum_x = self._config.canonical_field_width - 1
        maximum_y = self._config.canonical_field_height - 1
        if not 0 <= canonical_x <= maximum_x or not 0 <= canonical_y <= maximum_y:
            reasons.append("outside_canonical_field")
        edge_margin = self._config.field_edge_margin_canonical
        if (
            canonical_x < edge_margin
            or canonical_x > maximum_x - edge_margin
            or canonical_y < edge_margin
            or canonical_y > maximum_y - edge_margin
        ):
            reasons.append("near_field_edge")
        return candidate, reasons

    def _field_mask(self, shape: tuple[int, int], field_geometry: FieldGeometry) -> np.ndarray:
        """Build an inset pixel mask from the calibrated field polygon."""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        corners = np.asarray(field_geometry.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [corners], 255)
        inset = self._config.field_inset_pixels
        if inset <= 0:
            return mask
        kernel_size = inset * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(mask, kernel, iterations=1)
