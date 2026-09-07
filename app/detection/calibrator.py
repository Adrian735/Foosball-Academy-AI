"""Field-only static calibration for submitted videos."""

import uuid
from pathlib import Path

import cv2

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.table_contracts import TableCalibration
from app.detection.debug_renderer import render_field_detection, write_debug_image
from app.detection.field_consensus import FieldConsensus, FieldConsensusError
from app.detection.field_detector import FieldDetector
from app.detection.startup_frames import StartupFrameReader


class TableCalibrator:
    """Build a static field calibration without depending on the web stack."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a calibrator with the shared startup, detector, and consensus config."""
        self._config = config
        self._frame_reader = StartupFrameReader(config)
        self._field_detector = FieldDetector(config)
        self._field_consensus = FieldConsensus(config)

    def calibrate_field(self, video_path: str) -> TableCalibration:
        """Detect and combine playable-field geometry from a video startup window.

        Video validation errors are propagated to the caller. A video with no
        stable field consensus returns a report with ``field=None`` and a
        warning so callers can route it to review rather than treat it as a
        failed exercise.
        """
        startup = self._frame_reader.read(video_path)
        candidates = tuple(
            candidate
            for frame in startup.accepted_frames
            if (candidate := self._field_detector.detect(frame.image)) is not None
        )
        warnings: list[str] = []
        field = None
        if not candidates:
            warnings.append("No field candidate was detected in accepted startup frames")
        else:
            try:
                field = self._field_consensus.combine(candidates)
            except FieldConsensusError as error:
                warnings.append(str(error))

        confidence = field.confidence if field is not None else 0.0
        return TableCalibration(
            metadata=startup.metadata,
            sampled_frame_quality=startup.frame_quality,
            field=field,
            rods=(),
            confidence=confidence,
            warnings=tuple(warnings),
            detector_config_version=self._config.detector_config_version,
        )

    def write_field_debug_image(
        self,
        video_path: str,
        calibration: TableCalibration,
        output_directory: str,
    ) -> str:
        """Render the consensus field on the first accepted frame and save it.

        Returns the generated image path. The output is intentionally kept
        outside the calibration contract so debug artifacts never affect API
        or persistence consumers.
        """
        startup = self._frame_reader.read(video_path)
        if startup.accepted_frames:
            frame_index = startup.accepted_frames[0].frame_index
            frame_image = startup.accepted_frames[0].image
        else:
            frame_index, frame_image = self._read_first_frame(video_path)
        image = render_field_detection(
            frame_image,
            calibration.field,
            frame_index,
            ("No startup frame passed the quality gate", *calibration.warnings),
        )
        output_path = Path(output_directory) / f"field-{uuid.uuid4().hex}.png"
        write_debug_image(str(output_path), image)
        return str(output_path)

    @staticmethod
    def _read_first_frame(video_path: str) -> tuple[int, object]:
        """Read frame zero for diagnostics when all startup frames are rejected."""
        capture = cv2.VideoCapture(video_path)
        try:
            was_read, image = capture.read()
            if not was_read or image is None:
                raise ValueError("Cannot render field debug image: first frame is unreadable")
            return 0, image
        finally:
            capture.release()