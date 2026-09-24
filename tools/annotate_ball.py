"""Interactively create manually verified ball-tracking annotations."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

# Direct script execution places only ``tools/`` on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BallAnnotationSession:
    """Collect visible or hidden ball labels at selected video frames."""

    def __init__(
        self,
        video_path: str,
        tolerance_px: float,
        sample_every: int,
        max_frames: int,
    ) -> None:
        """Open a video and prepare a sparse annotation session."""
        self.video_path = video_path
        self.tolerance_px = tolerance_px
        if sample_every < 1:
            raise ValueError("sample_every must be at least 1")
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.sample_every = sample_every
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        self.source_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = min(self.source_frame_count, max_frames)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.samples: dict[int, dict[str, Any]] = {}
        self.frame_index = 0
        self.image: np.ndarray | None = None

    def close(self) -> None:
        """Release the video decoder."""
        self.capture.release()

    def read_frame(self, frame_index: int) -> np.ndarray:
        """Read and display one source frame without changing its index."""
        bounded_index = max(0, min(frame_index, self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, bounded_index)
        was_read, image = self.capture.read()
        if not was_read or image is None:
            raise ValueError(f"Unable to read frame {bounded_index}: {self.video_path}")
        self.frame_index = bounded_index
        self.image = image
        return image

    def annotate_visible(self, x_coordinate: int, y_coordinate: int) -> None:
        """Record a visible ball centre for the current frame."""
        self.samples[self.frame_index] = {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.frame_index / self.fps if self.fps > 0 else 0.0,
            "visible": True,
            "center": [x_coordinate, y_coordinate],
        }

    def annotate_hidden(self) -> None:
        """Record that the ball is not visibly annotatable in this frame."""
        self.samples[self.frame_index] = {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.frame_index / self.fps if self.fps > 0 else 0.0,
            "visible": False,
        }

    def undo(self) -> None:
        """Remove the annotation for the current frame."""
        self.samples.pop(self.frame_index, None)

    def fixture(self) -> dict[str, Any]:
        """Return the JSON-safe annotation document."""
        duration = self.frame_count / self.fps if self.fps > 0 else 0.0
        return {
            "schema_version": 1,
            "fixture": Path(self.video_path).name,
            "source_video": self.video_path,
            "video": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "frame_count": self.source_frame_count,
                "duration_seconds": duration,
            },
            "annotation": {
                "status": "human_verified",
                "coordinate_system": "source_pixels",
                "center_tolerance_px": self.tolerance_px,
                "sampling": {
                    "mode": "fixed_frame_stride",
                    "every_n_frames": self.sample_every,
                },
                "frame_limit": self.frame_count,
                "samples": [self.samples[index] for index in sorted(self.samples)],
            },
        }

    def preview(self) -> np.ndarray:
        """Render the current frame and annotation state for manual inspection."""
        if self.image is None:
            self.read_frame(self.frame_index)
        preview = self.image.copy()
        sample = self.samples.get(self.frame_index)
        if sample and sample["visible"]:
            center = sample["center"]
            cv2.circle(preview, (int(center[0]), int(center[1])), 12, (0, 255, 0), 2)
            cv2.putText(
                preview,
                "ball",
                (int(center[0]) + 14, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        status = "visible" if sample and sample["visible"] else "hidden" if sample else "unlabeled"
        cv2.putText(
            preview,
            f"frame {self.frame_index}/{self.frame_count - 1} | {status} | "
            f"stride={self.sample_every} | click=visible h=hidden "
            "left/right=step s=save u=undo q=quit",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview


def _parse_arguments() -> argparse.Namespace:
    """Parse the ball annotation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("output_path")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--start-position",
        choices=("beginning", "middle"),
        default="beginning",
        help="Start at the beginning or middle of the video; defaults to beginning",
    )
    parser.add_argument("--tolerance-px", type=float, default=12.0)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Annotate every Nth source frame; defaults to 10",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Limit the annotation window to this many source frames; defaults to 400",
    )
    return parser.parse_args()


def main() -> None:
    """Run the interactive annotation window and write a JSON fixture."""
    arguments = _parse_arguments()
    session = BallAnnotationSession(
        arguments.video_path,
        arguments.tolerance_px,
        arguments.sample_every,
        arguments.max_frames,
    )
    window_name = "Ball annotation"

    def handle_mouse(event: int, x_coordinate: int, y_coordinate: int, _: int, __: object) -> None:
        """Record a visible ball centre on a left click."""
        if event == cv2.EVENT_LBUTTONDOWN:
            session.annotate_visible(x_coordinate, y_coordinate)

    try:
        initial_frame = arguments.frame_index
        if arguments.start_position == "middle":
            initial_frame = session.frame_count // 2
        session.read_frame(initial_frame)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as error:
        session.close()
        raise SystemExit(
            "Interactive annotation requires GUI-enabled OpenCV and a display."
        ) from error

    cv2.setMouseCallback(window_name, handle_mouse)
    try:
        while True:
            cv2.imshow(window_name, session.preview())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("h"), ord("0")):
                session.annotate_hidden()
            elif key in (ord("u"), 8):
                session.undo()
            elif key in (ord("s"),):
                output_path = Path(arguments.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(session.fixture(), indent=2) + "\n", encoding="utf-8")
                print(f"Saved {len(session.samples)} samples to {output_path}")
                break
            elif key in (ord("q"), 27):
                break
            elif key in (81, ord("a")):
                session.read_frame(session.frame_index - session.sample_every)
            elif key in (83, ord("d")):
                session.read_frame(session.frame_index + session.sample_every)
            elif key == ord("n"):
                session.read_frame(session.frame_index + max(1, round(session.fps)))
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
