"""Tests for the event-analysis package boundary."""

import ast
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.ball_tracking.contracts import BallTrack
from app.detection.contracts.table_contracts import FieldGeometry, TableCalibration
from app.detection.contracts.video_contracts import VideoMetadata
from app.event_analysis import EventAnalysisInput


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVENT_ANALYSIS_ROOT = REPOSITORY_ROOT / "app" / "event_analysis"
FORBIDDEN_IMPORT_ROOTS = {
    "fastapi",
    "celery",
    "sqlalchemy",
    "app.api",
    "app.worker",
    "app.database",
    "app.storage",
    "database",
    "storage",
}


def _analysis_input() -> EventAnalysisInput:
    """Build the smallest valid event-analysis input from public CV contracts."""
    metadata = VideoMetadata("video.mp4", 30.0, 150, 1280, 720, 5.0)
    field = FieldGeometry(
        corners=((0.0, 0.0), (1000.0, 0.0), (1000.0, 600.0), (0.0, 600.0)),
        bounding_box=(0, 0, 1000, 600),
        confidence=0.9,
        detection_method="synthetic",
    )
    calibration = TableCalibration(metadata, (), field, (), 0.9)
    track = BallTrack(metadata.to_dict(), "calibration-1", "tracking-1", (), 0.0, 0.0, 0.0)
    return EventAnalysisInput(track=track, calibration=calibration)


def test_event_analysis_input_serializes_existing_tracking_and_calibration_contracts() -> None:
    """The boundary contract delegates serialization to existing CV contracts."""
    analysis_input = _analysis_input()

    payload = analysis_input.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["track"]["tracking_config_version"] == "tracking-1"
    assert payload["calibration"]["field"]["detection_method"] == "synthetic"


def test_event_analysis_input_is_immutable() -> None:
    """A caller cannot replace either input after constructing the contract."""
    analysis_input = _analysis_input()

    with pytest.raises(FrozenInstanceError):
        analysis_input.track = analysis_input.track


def test_event_analysis_package_has_no_infrastructure_imports() -> None:
    """Package source stays independent from service and persistence layers."""
    forbidden_imports: list[str] = []
    for source_path in EVENT_ANALYSIS_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            else:
                continue
            forbidden_imports.extend(
                name
                for name in imported_names
                if any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_IMPORT_ROOTS)
            )

    assert forbidden_imports == []


def test_importing_event_analysis_does_not_load_infrastructure_modules() -> None:
    """Importing the public boundary does not initialize service dependencies."""
    script = """
import sys
import app.event_analysis

for prefix in (
    "fastapi",
    "celery",
    "sqlalchemy",
    "app.api",
    "app.worker",
    "app.database",
    "app.storage",
    "database",
    "storage",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
