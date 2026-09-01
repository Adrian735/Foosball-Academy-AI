import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ExerciseType(str, enum.Enum):
    BALL_CONTROL_GOAL = "ball_control_goal"
    GOAL_STREAK = "goal_streak"
    PULL_SHOT = "pull_shot"
    SNAKE = "snake"
    INJECTION = "injection"


class SubmissionStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    current_level = Column(Integer, nullable=False, default=1)

    submissions = relationship("Submission", back_populates="user")


class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(String, primary_key=True, default=_uuid)
    level = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(Enum(ExerciseType), nullable=False)
    # rod/player position, min hold duration, target count, confidence thresholds, ...
    criteria = Column(JSON, nullable=False, default=dict)
    active = Column(Boolean, nullable=False, default=True)

    submissions = relationship("Submission", back_populates="exercise")


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    exercise_id = Column(String, ForeignKey("exercises.id"), nullable=False)
    video_url = Column(String, nullable=False)
    status = Column(Enum(SubmissionStatus), nullable=False, default=SubmissionStatus.QUEUED)
    metrics = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    decision_note = Column(Text, nullable=True)

    user = relationship("User", back_populates="submissions")
    exercise = relationship("Exercise", back_populates="submissions")
