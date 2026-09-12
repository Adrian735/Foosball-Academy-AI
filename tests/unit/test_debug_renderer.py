"""Tests for table-calibration debug overlays."""

import cv2
import numpy as np

from app.detection.contracts.rod_contracts import Rod, RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry, TableCalibration
from app.detection.contracts.video_contracts import VideoMetadata
from app.detection.debug_renderer import (
    render_expected_annotation,
    render_rod_candidates,
    render_table_calibration,
    write_debug_image,
)


def test_expected_annotation_renders_field_and_rod_lines() -> None:
    """Expected-fixture geometry produces a visible overlay without mutating input."""
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    expected = {
        "field_corners": [[10, 10], [110, 10], [110, 90], [10, 90]],
        "rod_lines": [[[10, 50], [110, 50]]],
    }

    rendered = render_expected_annotation(image, expected)

    assert np.array_equal(image, np.zeros_like(image))
    assert rendered[10, 60].tolist() == [0, 255, 255]
    assert rendered[50, 60].tolist() == [255, 0, 255]


def test_calibration_overlay_includes_field_rods_and_status(tmp_path) -> None:
    """A complete calibration can be rendered and encoded as a PNG artifact."""
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    calibration = TableCalibration(
        metadata=VideoMetadata("example.mp4", 30.0, 150, 120, 100, 5.0),
        sampled_frame_quality=(),
        field=FieldGeometry(((10.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)), (10, 10, 100, 80), 0.9, "manual"),
        rods=(Rod(0, ((10.0, 50.0), (110.0, 50.0)), 0.5, 0.9, 0.8),),
        confidence=0.9,
        warnings=("manual fixture",),
    )

    output_path = tmp_path / "overlay.png"
    rendered = render_table_calibration(image, calibration)
    write_debug_image(str(output_path), rendered)

    assert output_path.exists()
    assert cv2.imread(str(output_path)) is not None
    assert rendered[50, 60].tolist() == [255, 0, 255]


def test_rod_candidate_overlay_distinguishes_accepted_and_rejected_lines() -> None:
    """Candidate diagnostics render accepted magenta and rejected red lines."""
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    accepted = RodCandidate(((10.0, 30.0), (110.0, 30.0)), 0.5, 0.9, 0.2, 0.8)
    rejected = RodCandidate(((10.0, 70.0), (110.0, 70.0)), 0.8, 0.2, 0.0, 0.2)

    rendered = render_rod_candidates(image, (accepted,), (rejected,))

    assert rendered[30, 60].tolist() == [255, 0, 255]
    assert rendered[70, 60][0] == 0
    assert rendered[70, 60][1] == 0
    assert rendered[70, 60][2] > 200