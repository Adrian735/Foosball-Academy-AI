"""Strict loading and validation for persisted table calibration reports."""

import json
import math
from pathlib import Path
from typing import Any, Mapping

from app.detection.contracts.rod_contracts import Rod
from app.detection.contracts.table_contracts import FieldGeometry, TableCalibration
from app.detection.contracts.video_contracts import FrameQuality, VideoMetadata


class CalibrationLoadError(ValueError):
    """Raised when a calibration report is malformed or unsafe to track."""


def load_table_calibration(path: str | Path) -> TableCalibration:
    """Load and strictly validate a JSON-serialized table calibration."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationLoadError(f"Unable to read calibration JSON: {source}") from error
    return table_calibration_from_dict(payload)


def table_calibration_from_dict(payload: Mapping[str, Any]) -> TableCalibration:
    """Convert an exact calibration mapping into its immutable contract."""
    _require_mapping(payload, "calibration")
    _check_keys(
        payload,
        {"metadata", "sampled_frame_quality", "field", "rods", "confidence", "warnings", "detector_config_version"},
        "calibration",
    )
    metadata_data = _mapping(payload["metadata"], "metadata")
    _check_keys(
        metadata_data,
        {"path", "frames_per_second", "frame_count", "width", "height", "duration_seconds"},
        "metadata",
    )
    metadata = VideoMetadata(
        path=_string(metadata_data["path"], "metadata.path"),
        frames_per_second=_positive_float(metadata_data["frames_per_second"], "metadata.frames_per_second"),
        frame_count=_positive_int(metadata_data["frame_count"], "metadata.frame_count"),
        width=_positive_int(metadata_data["width"], "metadata.width"),
        height=_positive_int(metadata_data["height"], "metadata.height"),
        duration_seconds=_positive_float(metadata_data["duration_seconds"], "metadata.duration_seconds"),
    )

    quality_items = _sequence(payload["sampled_frame_quality"], "sampled_frame_quality")
    qualities = tuple(_frame_quality(_mapping(item, "sampled_frame_quality item")) for item in quality_items)
    field = _field_geometry(payload["field"])
    rods = tuple(_rod(_mapping(item, "rod")) for item in _sequence(payload["rods"], "rods"))
    return TableCalibration(
        metadata=metadata,
        sampled_frame_quality=qualities,
        field=field,
        rods=rods,
        confidence=_unit_float(payload["confidence"], "confidence"),
        warnings=tuple(_string(item, "warnings item") for item in _sequence(payload["warnings"], "warnings")),
        detector_config_version=_string(payload["detector_config_version"], "detector_config_version"),
    )


def require_trackable_calibration(calibration: TableCalibration, minimum_confidence: float = 0.60) -> TableCalibration:
    """Reject calibration that cannot safely constrain ball detection."""
    if calibration.field is None:
        raise CalibrationLoadError("Calibration has no field geometry")
    if calibration.warnings:
        raise CalibrationLoadError("Calibration requires review: " + "; ".join(calibration.warnings))
    if calibration.confidence < minimum_confidence:
        raise CalibrationLoadError(
            f"Calibration confidence {calibration.confidence:.3f} is below {minimum_confidence:.3f}"
        )
    return calibration


def _field_geometry(value: Any) -> FieldGeometry | None:
    if value is None:
        return None
    data = _mapping(value, "field")
    _check_keys(data, {"corners", "bounding_box", "confidence", "detection_method"}, "field")
    corners = _points(data["corners"], 4, "field.corners")
    bounding_box = tuple(_number(item, "field.bounding_box item") for item in _sequence(data["bounding_box"], "field.bounding_box"))
    if len(bounding_box) != 4:
        raise CalibrationLoadError("field.bounding_box must contain four numbers")
    return FieldGeometry(
        corners=corners,  # type: ignore[arg-type]
        bounding_box=tuple(int(item) for item in bounding_box),
        confidence=_unit_float(data["confidence"], "field.confidence"),
        detection_method=_string(data["detection_method"], "field.detection_method"),
    )


def _frame_quality(data: Mapping[str, Any]) -> FrameQuality:
    _check_keys(data, {"frame_index", "timestamp_seconds", "blur_score", "brightness_score", "accepted", "rejection_reason"}, "frame quality")
    rejection_reason = data["rejection_reason"]
    if rejection_reason is not None and not isinstance(rejection_reason, str):
        raise CalibrationLoadError("frame quality rejection_reason must be a string or null")
    if not isinstance(data["accepted"], bool):
        raise CalibrationLoadError("frame quality accepted must be boolean")
    return FrameQuality(
        frame_index=_nonnegative_int(data["frame_index"], "frame_index"),
        timestamp_seconds=_nonnegative_float(data["timestamp_seconds"], "timestamp_seconds"),
        blur_score=_nonnegative_float(data["blur_score"], "blur_score"),
        brightness_score=_nonnegative_float(data["brightness_score"], "brightness_score"),
        accepted=data["accepted"],
        rejection_reason=rejection_reason,
    )


def _rod(data: Mapping[str, Any]) -> Rod:
    _check_keys(data, {"index", "line", "field_relative_y", "confidence", "player_colour_evidence"}, "rod")
    return Rod(
        index=_nonnegative_int(data["index"], "rod.index"),
        line=_points(data["line"], 2, "rod.line"),  # type: ignore[arg-type]
        field_relative_y=_number(data["field_relative_y"], "rod.field_relative_y"),
        confidence=_unit_float(data["confidence"], "rod.confidence"),
        player_colour_evidence=_unit_float(data["player_colour_evidence"], "rod.player_colour_evidence"),
    )


def _check_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise CalibrationLoadError(f"Invalid {name}: {'; '.join(details)}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationLoadError(f"{name} must be an object")
    return value


def _require_mapping(value: Any, name: str) -> None:
    _mapping(value, name)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CalibrationLoadError(f"{name} must be an array")
    return value


def _points(value: Any, expected_length: int, name: str) -> tuple[tuple[float, float], ...]:
    points = _sequence(value, name)
    if len(points) != expected_length:
        raise CalibrationLoadError(f"{name} must contain {expected_length} points")
    result = []
    for index, point in enumerate(points):
        values = _sequence(point, f"{name}[{index}]")
        if len(values) != 2:
            raise CalibrationLoadError(f"{name}[{index}] must contain two numbers")
        result.append((_number(values[0], f"{name}[{index}][0]"), _number(values[1], f"{name}[{index}][1]")))
    return tuple(result)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CalibrationLoadError(f"{name} must be a finite number")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise CalibrationLoadError(f"{name} must be greater than zero")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise CalibrationLoadError(f"{name} must not be negative")
    return result


def _unit_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0 <= result <= 1:
        raise CalibrationLoadError(f"{name} must be between zero and one")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise CalibrationLoadError(f"{name} must be greater than zero")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationLoadError(f"{name} must be a non-negative integer")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise CalibrationLoadError(f"{name} must be a string")
    return value
