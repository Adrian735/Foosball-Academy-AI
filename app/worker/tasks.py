from app.database import SessionLocal
from app.models import Submission, SubmissionStatus
from app.worker.celery_app import celery_app


@celery_app.task(name="app.worker.tasks.process_submission")
def process_submission(submission_id: str) -> None:
    """Runs the CV pipeline for one submission and records the outcome.

    The field/ball/rod detection and event-analysis modules aren't wired in
    yet (see docs/PLAN.md Phase 1) -- this task only manages the status
    lifecycle so the API/queue/worker plumbing can be exercised end to end.
    """
    db = SessionLocal()
    try:
        submission = db.get(Submission, submission_id)
        if submission is None:
            return

        submission.status = SubmissionStatus.PROCESSING
        db.commit()

        # TODO(Phase 1): field_detector -> ball_tracker -> rod_tracker
        # -> event_analyzer -> validators/<exercise_type>.py
        submission.status = SubmissionStatus.PENDING_REVIEW
        db.commit()
    finally:
        db.close()
