"""API coverage for the one-call ball-tracking development endpoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import detection
from app.storage import VideoStorage


def test_ball_tracking_endpoint_runs_pipeline_and_debug_output(tmp_path, monkeypatch):
    """One uploaded video produces a track report and optional debug frames."""
    api = FastAPI()
    api.include_router(detection.router)
    monkeypatch.setattr(detection, "storage", VideoStorage(str(tmp_path / "videos")))
    monkeypatch.setattr(detection.settings, "debug_output_dir", str(tmp_path / "debug"))

    video_path = Path("tests/table-detection_tests/table-detection_test-1.mp4")
    with video_path.open("rb") as video:
        response = TestClient(api).post(
            "/detection/ball-tracking?debug=true",
            files={"video": (video_path.name, video, "video/mp4")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["calibration"]["field"] is not None
    assert result["ball_tracking"]["video_metadata"]["frame_count"] == 162
    assert result["debug"]["frame_count"] == 162
    assert len(list((tmp_path / "debug" / "ball-tracking").glob("frame-*.png"))) == 162