"""Development endpoints for running isolated detection stages."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ball_tracking.calibration import CalibrationLoadError, require_trackable_calibration
from app.ball_tracking.contracts import BallTrackingResult
from app.ball_tracking.debug_renderer import render_ball_tracking_frame, write_debug_image
from app.ball_tracking.frame_reader import BallVideoReadError, SequentialFrame, SequentialFrameReader
from app.ball_tracking.tracker import BallTracker
from app.config import settings
from app.detection.calibrator import TableCalibrator
from app.detection.contracts.table_contracts import FieldGeometry
from app.detection.startup_frames import VideoValidationError
from app.storage import storage

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("/field")
async def detect_field(
    request: Request,
    video: UploadFile | None = File(None),
) -> dict:
    """Accept multipart or raw video and return field-only diagnostics."""
    if video is not None:
        video_path = storage.save(video)
    else:
        content = await request.body()
        if not content:
            raise HTTPException(status_code=422, detail="A video upload is required")
        video_path = storage.save_bytes(content)

    calibrator = TableCalibrator()
    try:
        calibration = calibrator.calibrate_field(video_path)
        result = calibration.to_dict()
        if settings.image_debug:
            result["debug_image_path"] = calibrator.write_field_debug_image(
                video_path,
                calibration,
                settings.debug_output_dir,
            )
        return result
    except VideoValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/ball-tracking")
async def track_ball(
    request: Request,
    video: UploadFile | None = File(None),
    debug: bool = False,
) -> dict:
    """Calibrate, track, and optionally render one video in a single call."""
    if video is not None:
        video_path = storage.save(video)
    else:
        content = await request.body()
        if not content:
            raise HTTPException(status_code=422, detail="A video upload is required")
        video_path = storage.save_bytes(content)

    try:
        calibration = TableCalibrator().calibrate_field(video_path)
        calibration = require_trackable_calibration(calibration)
        if calibration.field is None:
            raise CalibrationLoadError("Calibration has no field geometry")
        read_result = SequentialFrameReader().read(video_path)
        tracking_result = BallTracker().track(
            read_result.frames,
            calibration.field,
            read_result.metadata,
            calibration.detector_config_version,
            read_result.warnings,
        )
        response = {
            "calibration": calibration.to_dict(),
            "ball_tracking": tracking_result.to_dict(),
        }
        if debug:
            debug_dir = _write_ball_tracking_debug(
                read_result.frames,
                calibration.field,
                tracking_result,
            )
            response["debug"] = {
                "output_dir": str(debug_dir),
                "frame_count": len(read_result.frames),
            }
        return response
    except VideoValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (CalibrationLoadError, BallVideoReadError, OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _write_ball_tracking_debug(
    frames: tuple[SequentialFrame, ...],
    field: FieldGeometry,
    tracking_result: BallTrackingResult,
) -> Path:
    """Write per-frame ball-tracking overlays and return their directory."""
    output_dir = Path(settings.debug_output_dir) / "ball-tracking"
    observations = tracking_result.track.observations
    for index, frame in enumerate(frames):
        previous = observations[index - 1] if index else None
        rendered = render_ball_tracking_frame(
            frame.image,
            field,
            observations[index],
            tracking_result.candidates_by_frame[index],
            previous,
        )
        write_debug_image(output_dir / f"frame-{frame.frame_index:06d}.png", rendered)
    return output_dir