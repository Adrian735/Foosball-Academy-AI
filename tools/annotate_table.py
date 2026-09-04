"""Interactively create a manually verified table-detection expected fixture."""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

# Direct script execution places only `tools/` on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detection.debug_renderer import render_expected_annotation, write_debug_image


class TableAnnotationSession:
    """Collect ordered field corners and rod line endpoints from mouse clicks."""

    def __init__(self, image: np.ndarray) -> None:
        """Create an annotation session for the selected BGR video frame."""
        self.image = image
        self.field_corners: list[list[int]] = []
        self.rod_lines: list[list[list[int]]] = []
        self._pending_rod_start: Optional[list[int]] = None

    def add_click(self, x_coordinate: int, y_coordinate: int) -> None:
        """Record a corner first, then pair subsequent clicks into rod segments."""
        point = [x_coordinate, y_coordinate]
        if len(self.field_corners) < 4:
            self.field_corners.append(point)
        elif self._pending_rod_start is None:
            self._pending_rod_start = point
        else:
            self.rod_lines.append([self._pending_rod_start, point])
            self._pending_rod_start = None

    def undo(self) -> None:
        """Remove the latest incomplete point, rod, or field corner."""
        if self._pending_rod_start is not None:
            self._pending_rod_start = None
        elif self.rod_lines:
            self.rod_lines.pop()
        elif self.field_corners:
            self.field_corners.pop()

    def expected_fixture(self, video_path: str) -> dict[str, object]:
        """Build the documented expected-fixture mapping from recorded clicks."""
        return {
            "fixture": Path(video_path).name,
            "expected_rod_count": len(self.rod_lines),
            "field_corners": self.field_corners,
            "corner_tolerance_px": 25,
            "rod_lines": self.rod_lines,
            "rod_y_positions": [
                round((line[0][1] + line[1][1]) / 2.0, 2) for line in self.rod_lines
            ],
            "rod_tolerance_px": 18,
        }

    def preview(self, video_path: str) -> np.ndarray:
        """Render the current manual annotations over the selected frame."""
        return render_expected_annotation(self.image, self.expected_fixture(video_path))


def _read_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Decode one BGR frame from a video for manual inspection."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        was_read, image = capture.read()
        if not was_read or image is None:
            raise ValueError(f"Unable to read frame {frame_index}: {video_path}")
        return image
    finally:
        capture.release()


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for one interactive annotation session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("output_path")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--export-frame",
        help="Write the selected frame to this path and exit without opening a window",
    )
    return parser.parse_args()


def main() -> None:
    """Run the OpenCV annotation window and write JSON plus a PNG preview."""
    arguments = _parse_arguments()
    image = _read_frame(arguments.video_path, arguments.frame_index)
    if arguments.export_frame:
        write_debug_image(arguments.export_frame, image)
        print(f"Exported frame to {arguments.export_frame}")
        return

    session = TableAnnotationSession(image)
    window_name = "Table annotation"

    def handle_mouse(event: int, x_coordinate: int, y_coordinate: int, _: int, __: object) -> None:
        """Record left clicks and undo the latest action on right click."""
        if event == cv2.EVENT_LBUTTONDOWN:
            session.add_click(x_coordinate, y_coordinate)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.undo()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as error:
        raise SystemExit(
            "Interactive annotation requires a GUI-enabled OpenCV build and a display. "
            "The installed opencv-python-headless package cannot open windows. "
            "Run this command on a desktop environment after installing opencv-python, "
            "or use --export-frame to save a frame for external annotation."
        ) from error
    cv2.setMouseCallback(window_name, handle_mouse)
    while True:
        cv2.imshow(window_name, session.preview(arguments.video_path))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            if len(session.field_corners) != 4 or session._pending_rod_start is not None:
                raise ValueError("Record exactly four corners and complete every rod line before saving")
            fixture = session.expected_fixture(arguments.video_path)
            output_path = Path(arguments.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
            write_debug_image(str(output_path.with_suffix(".png")), session.preview(arguments.video_path))
            break
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()