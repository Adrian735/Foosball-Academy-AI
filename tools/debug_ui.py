"""Run local visual diagnostics for table calibration and ball tracking.

Launch with ``streamlit run tools/debug_ui.py`` from the repository root.
This developer tool deliberately calls CV modules directly; it does not create
submissions or exercise decisions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.ball_tracking.calibration import CalibrationLoadError, require_trackable_calibration
from app.ball_tracking.contracts import BallTrackingResult
from app.ball_tracking.debug_renderer import render_ball_tracking_frame
from app.ball_tracking.frame_reader import BallVideoReadError, SequentialFrame, SequentialFrameReader
from app.ball_tracking.tracker import BallTracker
from app.detection.calibrator import TableCalibrator
from app.detection.contracts.table_contracts import TableCalibration
from app.detection.debug_renderer import render_table_calibration
from app.detection.startup_frames import VideoValidationError


@dataclass(frozen=True)
class DebugAnalysis:
    """Serializable paths and reports produced by one local debug run."""

    calibration: dict[str, object]
    tracking: dict[str, object] | None
    annotated_video_path: Path | None
    tracking_skip_reason: str | None


class UploadedVideo(Protocol):
    """Minimal uploaded-file interface used by the Streamlit boundary."""

    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded file contents."""
        ...


def run_debug_analysis(video_path: Path, output_directory: Path) -> DebugAnalysis:
    """Calibrate one video, optionally track its ball, and write an overlay MP4.

    The calibration report is always returned. Ball tracking only runs after
    the same review gate used by the standalone tracking runner accepts the
    calibration.
    """
    calibration = TableCalibrator().calibrate(str(video_path))
    calibration_report = calibration.to_dict()
    try:
        require_trackable_calibration(calibration)
    except CalibrationLoadError as error:
        return DebugAnalysis(calibration_report, None, None, str(error))

    if calibration.field is None:
        return DebugAnalysis(
            calibration_report,
            None,
            None,
            "Calibration has no field geometry",
        )

    read_result = SequentialFrameReader().read(str(video_path))
    tracking_result = BallTracker().track(
        read_result.frames,
        calibration.field,
        read_result.metadata,
        calibration.detector_config_version,
        read_result.warnings,
    )
    output_path = output_directory / "annotated-debug.mp4"
    _write_annotated_video(
        output_path,
        calibration,
        read_result.metadata.frames_per_second,
        read_result.frames,
        tracking_result,
    )
    return DebugAnalysis(
        calibration_report,
        tracking_result.to_dict(),
        output_path,
        None,
    )


def _write_annotated_video(
    output_path: Path,
    calibration: TableCalibration,
    frames_per_second: float,
    frames: tuple[SequentialFrame, ...],
    tracking_result: BallTrackingResult,
) -> None:
    """Encode calibration and ball-tracking diagnostics into a playable MP4."""
    if not frames or calibration.field is None:
        raise ValueError("Cannot write debug video without frames and field geometry")

    first_frame = frames[0].image
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f"{output_path.stem}-intermediate.mp4")
    writer = cv2.VideoWriter(
        str(intermediate_path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        frames_per_second,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to open intermediate debug video output: {intermediate_path}")

    try:
        observations = tracking_result.track.observations
        candidates_by_frame = tracking_result.candidates_by_frame
        for index, frame in enumerate(frames):
            overlay = render_table_calibration(frame.image, calibration)
            overlay = render_ball_tracking_frame(
                overlay,
                calibration.field,
                observations[index],
                candidates_by_frame[index],
                observations[index - 1] if index else None,
            )
            writer.write(overlay)
    finally:
        writer.release()
    _transcode_for_browser_preview(intermediate_path, output_path)
    intermediate_path.unlink(missing_ok=True)


def _transcode_for_browser_preview(intermediate_path: Path, output_path: Path) -> None:
    """Transcode an OpenCV MP4 into the H.264 format browsers can preview."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise ValueError("FFmpeg is required to create a browser-previewable debug video")
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(intermediate_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip().splitlines()
        reason = details[-1] if details else "unknown FFmpeg error"
        raise ValueError(f"Unable to encode browser-previewable debug video: {reason}") from error


def _write_uploaded_video(uploaded_video: UploadedVideo) -> Path:
    """Persist one uploaded video into an isolated temporary directory."""
    output_directory = Path(tempfile.mkdtemp(prefix="foosball-debug-"))
    video_path = output_directory / Path(uploaded_video.name).name
    video_path.write_bytes(uploaded_video.getvalue())
    return video_path


def _render_analysis(analysis: DebugAnalysis) -> None:
    """Render reports, warnings, and downloads for a completed debug run."""
    calibration_column, tracking_column = st.columns(2)
    with calibration_column:
        st.subheader("Field and Rod Detection")
        st.json(analysis.calibration)
    with tracking_column:
        st.subheader("Ball Tracking")
        if analysis.tracking is None:
            st.warning(f"Skipped: {analysis.tracking_skip_reason}")
        else:
            st.json(analysis.tracking)

    st.download_button(
        "Download calibration report",
        json.dumps(analysis.calibration, indent=2),
        file_name="calibration.json",
        mime="application/json",
    )
    if analysis.tracking is not None:
        st.download_button(
            "Download ball tracking report",
            json.dumps(analysis.tracking, indent=2),
            file_name="ball-track.json",
            mime="application/json",
        )
    if analysis.annotated_video_path is not None:
        st.subheader("Annotated Debug Video")
        video_bytes = analysis.annotated_video_path.read_bytes()
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download annotated video",
            video_bytes,
            file_name="annotated-debug.mp4",
            mime="video/mp4",
        )


def main() -> None:
    """Render the Streamlit developer interface for local CV inspection."""
    st.set_page_config(page_title="Foosball CV Debug", layout="wide")
    st.title("Foosball CV Debug")
    uploaded_video = st.file_uploader("Video", type=["mp4", "mov", "mkv", "avi"])
    if uploaded_video is None:
        return

    st.video(uploaded_video)
    if st.button("Run field, rod, and ball diagnostics", type="primary"):
        with st.spinner("Calibrating table and tracking ball..."):
            try:
                video_path = _write_uploaded_video(uploaded_video)
                analysis = run_debug_analysis(video_path, video_path.parent)
                st.session_state["debug_analysis"] = analysis
            except (BallVideoReadError, VideoValidationError, OSError, ValueError) as error:
                st.session_state.pop("debug_analysis", None)
                st.error(str(error))

    analysis = st.session_state.get("debug_analysis")
    if isinstance(analysis, DebugAnalysis):
        _render_analysis(analysis)


if __name__ == "__main__":
    main()