"""Tests for the static-calibration worker integration."""

from types import SimpleNamespace

from app.ball_tracking.contracts import BallTrack, BallTrackingResult
from app.detection.contracts.rod_contracts import Rod
from app.detection.contracts.table_contracts import FieldGeometry, TableCalibration
from app.detection.contracts.video_contracts import VideoMetadata
from app.detection.startup_frames import VideoValidationError
from app.models import SubmissionStatus
from app.worker import tasks


class _Session:
    """Minimal database-session double that records worker lifecycle calls."""

    def __init__(self, submission: SimpleNamespace | None) -> None:
        """Create a session double returning the supplied submission."""
        self._submission = submission
        self.commit_count = 0
        self.closed = False

    def get(self, _model: object, _submission_id: str) -> SimpleNamespace | None:
        """Return the configured submission regardless of ORM lookup details."""
        return self._submission

    def commit(self) -> None:
        """Record a completed persistence boundary."""
        self.commit_count += 1

    def close(self) -> None:
        """Record worker cleanup."""
        self.closed = True


class _Calibrator:
    """Calibration-facade double returning a configured result or error."""

    def __init__(self, outcome: TableCalibration | VideoValidationError) -> None:
        """Store the configured facade outcome."""
        self._outcome = outcome

    def calibrate(self, _video_path: str) -> TableCalibration:
        """Return the calibration report or raise the expected validation error."""
        if isinstance(self._outcome, VideoValidationError):
            raise self._outcome
        return self._outcome


class _Reader:
    """Frame-reader double returning a prepared decoded input."""

    def __init__(self, metadata: VideoMetadata) -> None:
        """Store metadata for the synthetic read result."""
        self._result = SimpleNamespace(metadata=metadata, frames=(), warnings=())

    def read(self, _video_path: str) -> SimpleNamespace:
        """Return the prepared empty frame sequence."""
        return self._result


class _Tracker:
    """Tracker double recording the calibration geometry it receives."""

    def __init__(self, result: BallTrackingResult) -> None:
        """Store the synthetic tracking result."""
        self.result = result
        self.called = False

    def track(self, *_args: object, **_kwargs: object) -> BallTrackingResult:
        """Return the result and record that tracking was invoked."""
        self.called = True
        return self.result


def _calibration(
    confidence: float = 0.9,
    warnings: tuple[str, ...] = (),
    rod_count: int = 8,
) -> TableCalibration:
    """Create a JSON-safe static calibration report for worker tests."""
    field = FieldGeometry(
        corners=((0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)),
        bounding_box=(0, 0, 100, 60),
        confidence=confidence,
        detection_method="test",
    )
    rods = tuple(
        Rod(
            index=index,
            line=((0.0, float(index)), (100.0, float(index))),
            field_relative_y=index / 10.0,
            confidence=confidence,
            player_colour_evidence=0.5,
        )
        for index in range(rod_count)
    )
    return TableCalibration(
        metadata=VideoMetadata("video.mp4", 30.0, 150, 1280, 720, 5.0),
        sampled_frame_quality=(),
        field=field,
        rods=rods,
        confidence=confidence,
        warnings=warnings,
        detector_config_version="test",
    )


def _submission() -> SimpleNamespace:
    """Create the worker-owned fields of a queued submission."""
    return SimpleNamespace(
        video_url="video.mp4",
        status=SubmissionStatus.QUEUED,
        metrics=None,
        confidence=None,
    )


def test_process_submission_persists_clean_calibration(monkeypatch) -> None:
    """A clean calibration and ball track are both persisted for review."""
    submission = _submission()
    session = _Session(submission)
    calibration = _calibration()
    tracking = BallTrackingResult(
        BallTrack(calibration.metadata.to_dict(), "test", "1", (), 0.9, 0.0, 0.8)
    )
    tracker = _Tracker(tracking)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(tasks, "TableCalibrator", lambda: _Calibrator(calibration))
    monkeypatch.setattr(tasks, "SequentialFrameReader", lambda: _Reader(calibration.metadata))
    monkeypatch.setattr(tasks, "BallTracker", lambda: tracker)

    tasks.process_submission.run("submission-id")

    assert submission.status is SubmissionStatus.PENDING_REVIEW
    assert submission.confidence == tracking.track.confidence
    assert submission.metrics == {
        "calibration": calibration.to_dict(),
        "calibration_requires_review": False,
        "ball_tracking": tracking.to_dict(),
        "ball_tracking_requires_review": False,
    }
    assert tracker.called
    assert session.commit_count == 2
    assert session.closed


def test_process_submission_routes_calibration_warnings_to_review(monkeypatch) -> None:
    """A structured calibration warning is persisted as a review requirement."""
    submission = _submission()
    session = _Session(submission)
    calibration = _calibration(warnings=("Rod consensus requires review: incomplete layout",), rod_count=0)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(tasks, "TableCalibrator", lambda: _Calibrator(calibration))
    tracker = _Tracker(BallTrackingResult(BallTrack({}, "", "", (), 0.0, 0.0, 0.0)))
    monkeypatch.setattr(tasks, "BallTracker", lambda: tracker)

    tasks.process_submission.run("submission-id")

    assert submission.status is SubmissionStatus.PENDING_REVIEW
    assert submission.confidence == calibration.confidence
    assert submission.metrics["calibration"] == calibration.to_dict()
    assert submission.metrics["calibration_requires_review"] is True
    assert submission.metrics["ball_tracking"] is None
    assert submission.metrics["ball_tracking_skipped"] == "calibration_requires_review"
    assert not tracker.called
    assert session.commit_count == 2
    assert session.closed


def test_process_submission_keeps_low_confidence_track_pending(monkeypatch) -> None:
    """A low-confidence track is persisted but never auto-approved or rejected."""
    submission = _submission()
    session = _Session(submission)
    calibration = _calibration()
    tracking = BallTrackingResult(
        BallTrack(
            calibration.metadata.to_dict(),
            "test",
            "1",
            (),
            0.2,
            1.0,
            0.2,
            ("track_coverage_below_threshold",),
        )
    )
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(tasks, "TableCalibrator", lambda: _Calibrator(calibration))
    monkeypatch.setattr(tasks, "SequentialFrameReader", lambda: _Reader(calibration.metadata))
    monkeypatch.setattr(tasks, "BallTracker", lambda: _Tracker(tracking))

    tasks.process_submission.run("submission-id")

    assert submission.status is SubmissionStatus.PENDING_REVIEW
    assert submission.confidence == 0.2
    assert submission.metrics["ball_tracking_requires_review"] is True
    assert submission.status not in (SubmissionStatus.APPROVED, SubmissionStatus.REJECTED)


def test_process_submission_persists_invalid_video_as_structured_failure(monkeypatch) -> None:
    """An invalid video becomes a reviewable zero-confidence calibration failure."""
    submission = _submission()
    session = _Session(submission)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        tasks,
        "TableCalibrator",
        lambda: _Calibrator(VideoValidationError("Video resolution is below the supported minimum")),
    )

    tasks.process_submission.run("submission-id")

    assert submission.status is SubmissionStatus.PENDING_REVIEW
    assert submission.confidence == 0.0
    assert submission.metrics == {
        "calibration": None,
        "calibration_failure": {
            "type": "video_validation",
            "message": "Video resolution is below the supported minimum",
        },
    }
    assert session.commit_count == 2
    assert session.closed


def test_process_submission_closes_the_session_when_submission_is_missing(monkeypatch) -> None:
    """A missing submission returns without leaving the database session open."""
    session = _Session(None)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)

    tasks.process_submission.run("missing-submission-id")

    assert session.commit_count == 0
    assert session.closed