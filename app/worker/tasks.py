from app.database import SessionLocal
from app.detection.calibrator import TableCalibrator
from app.detection.config import DEFAULT_DETECTION_CONFIG
from app.detection.contracts.table_contracts import TableCalibration
from app.detection.startup_frames import VideoValidationError
from app.models import Submission, SubmissionStatus
from app.worker.celery_app import celery_app


@celery_app.task(name="app.worker.tasks.process_submission")
def process_submission(submission_id: str) -> None:
    """Runs the CV pipeline for one submission and records the outcome.

    Static calibration is the only CV stage wired here. Every submission stays
    pending review until ball tracking and exercise validation are available.
    """
    db = SessionLocal()
    try:
        submission = db.get(Submission, submission_id)
        if submission is None:
            return

        submission.status = SubmissionStatus.PROCESSING
        db.commit()

        try:
            calibration = TableCalibrator().calibrate(submission.video_url)
        except VideoValidationError as error:
            submission.metrics = {
                "calibration": None,
                "calibration_failure": {
                    "type": "video_validation",
                    "message": str(error),
                },
            }
            submission.confidence = 0.0
            submission.status = SubmissionStatus.PENDING_REVIEW
            db.commit()
            return

        submission.metrics = {
            "calibration": calibration.to_dict(),
            "calibration_requires_review": _calibration_requires_review(calibration),
        }
        submission.confidence = calibration.confidence
        submission.status = SubmissionStatus.PENDING_REVIEW
        db.commit()
    finally:
        db.close()


def _calibration_requires_review(calibration: TableCalibration) -> bool:
    """Return whether static-calibration diagnostics require human review."""
    return (
        calibration.field is None
        or not calibration.rods
        or bool(calibration.warnings)
        or calibration.confidence < DEFAULT_DETECTION_CONFIG.minimum_calibration_confidence
    )
