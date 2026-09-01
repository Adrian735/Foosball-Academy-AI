from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from app.models import ExerciseType, SubmissionStatus


class ExerciseCreate(BaseModel):
    level: int
    name: str
    description: Optional[str] = None
    type: ExerciseType
    criteria: dict[str, Any] = {}
    active: bool = True


class ExerciseOut(ExerciseCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    exercise_id: str
    video_url: str
    status: SubmissionStatus
    metrics: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
