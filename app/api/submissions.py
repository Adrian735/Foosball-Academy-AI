from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Exercise, Submission, SubmissionStatus
from app.schemas import SubmissionOut
from app.storage import storage
from app.worker.tasks import process_submission

router = APIRouter(prefix="/submissions", tags=["submissions"])


@router.post("", response_model=SubmissionOut, status_code=202)
def create_submission(
    user_id: str = Form(...),
    exercise_id: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    exercise = db.get(Exercise, exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="Exercise not found")

    video_path = storage.save(video)

    submission = Submission(
        user_id=user_id,
        exercise_id=exercise_id,
        video_url=video_path,
        status=SubmissionStatus.QUEUED,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    process_submission.delay(submission.id)

    return submission


@router.get("/{submission_id}", response_model=SubmissionOut)
def get_submission(submission_id: str, db: Session = Depends(get_db)):
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission
