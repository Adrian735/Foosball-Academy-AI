from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import detection
from app.storage import VideoStorage


def test_field_detection_upload_returns_field_only_result(tmp_path, monkeypatch):
    """The development endpoint accepts a video and returns field calibration."""
    api = FastAPI()
    api.include_router(detection.router)
    monkeypatch.setattr(detection, "storage", VideoStorage(str(tmp_path)))

    video_path = Path("tests/table-detection_tests/table-detection_test-1.mp4")
    with video_path.open("rb") as video:
        response = TestClient(api).post(
            "/detection/field",
            files={"video": (video_path.name, video, "video/mp4")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["field"]["detection_method"] == "frame_consensus"
    assert result["rods"] == []
    assert len(result["sampled_frame_quality"]) == 15
    assert result["confidence"] > 0.6


def test_field_detection_debug_mode_exports_rendered_field_and_prints_rods(tmp_path, monkeypatch, capsys):
    """Debug mode writes one field overlay for an endpoint request."""
    api = FastAPI()
    api.include_router(detection.router)
    monkeypatch.setattr(detection, "storage", VideoStorage(str(tmp_path / "videos")))
    monkeypatch.setattr(detection.settings, "image_debug", True)
    monkeypatch.setattr(detection.settings, "debug_output_dir", str(tmp_path / "debug"))

    video_path = Path("tests/table-detection_tests/table-detection_test-1.mp4")
    with video_path.open("rb") as video:
        response = TestClient(api).post(
            "/detection/field",
            files={"video": (video_path.name, video, "video/mp4")},
        )

    assert response.status_code == 200
    exports = list((tmp_path / "debug").glob("field-*.png"))
    assert len(exports) == 1
    assert response.json()["debug_image_path"] == str(exports[0])
    debug_output = capsys.readouterr().out
    assert "[rod-debug] frame=" in debug_output
    assert "accepted=" in debug_output
    assert "rejected=" in debug_output


def test_field_detection_accepts_raw_video_body(tmp_path, monkeypatch):
    """The endpoint also accepts Postman's raw video/mp4 request format."""
    api = FastAPI()
    api.include_router(detection.router)
    monkeypatch.setattr(detection, "storage", VideoStorage(str(tmp_path)))

    video_path = Path("tests/table-detection_tests/table-detection_test-1.mp4")
    response = TestClient(api).post(
        "/detection/field",
        content=video_path.read_bytes(),
        headers={"Content-Type": "video/mp4"},
    )

    assert response.status_code == 200
    assert response.json()["field"]["detection_method"] == "frame_consensus"


def test_field_detection_debug_mode_exports_when_all_frames_fail_quality(
    tmp_path,
    monkeypatch,
):
    """Debug mode still exports a diagnostic image when no frame is accepted."""
    api = FastAPI()
    api.include_router(detection.router)
    monkeypatch.setattr(detection, "storage", VideoStorage(str(tmp_path / "videos")))
    monkeypatch.setattr(detection.settings, "image_debug", True)
    monkeypatch.setattr(detection.settings, "debug_output_dir", str(tmp_path / "debug"))

    video_path = Path("tests/table-detection_tests/table-detection_test-1.mp4")
    original_calibrator = detection.TableCalibrator

    class RejectAllFrames:
        def __init__(self, reader):
            self._reader = reader

        def read(self, video_path):
            result = self._reader.read(video_path)
            return type(result)(result.metadata, result.frame_quality, tuple())

    class RejectingCalibrator(original_calibrator):
        def __init__(self):
            super().__init__()
            self._frame_reader = RejectAllFrames(self._frame_reader)

    monkeypatch.setattr(detection, "TableCalibrator", RejectingCalibrator)
    with video_path.open("rb") as video:
        response = TestClient(api).post(
            "/detection/field",
            files={"video": (video_path.name, video, "video/mp4")},
        )

    assert response.status_code == 200
    exports = list((tmp_path / "debug").glob("field-*.png"))
    assert len(exports) == 1