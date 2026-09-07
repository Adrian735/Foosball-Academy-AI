"""Development endpoints for running isolated detection stages."""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.detection.calibrator import TableCalibrator
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