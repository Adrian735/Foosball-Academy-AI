"""Configuration values for static Bonzini table detection."""

from dataclasses import dataclass
from typing import Tuple

HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass(frozen=True)
class DetectionConfig:
    """Immutable thresholds used by the table and rod detection pipeline.

    Values are field-relative where possible so later detector stages work
    across supported 720p-or-higher video resolutions.
    """

    # Identifies the exact threshold set used to produce a calibration result.
    detector_config_version: str = "2"
    # Limits static calibration to the first seconds of a submitted video.
    calibration_window_seconds: float = 3.0
    # Caps decoded startup frames before sampling to bound processing cost.
    maximum_calibration_frames: int = 90
    # Defines how many evenly spaced startup frames are evaluated.
    sampled_frame_count: int = 15
    # Requires enough usable frames before accepting a calibration.
    minimum_accepted_frames: int = 5
    # Rejects clips too short to provide a stable startup calibration window.
    minimum_video_duration_seconds: float = 5.0
    # Enforces the minimum supported horizontal video resolution.
    minimum_video_width: int = 1280
    # Enforces the minimum supported vertical video resolution.
    minimum_video_height: int = 720
    # Rejects frames whose Laplacian variance indicates excessive blur.
    minimum_laplacian_variance: float = 50.0
    # Rejects frames whose mean luminance is too dark for reliable detection.
    minimum_mean_luminance: float = 25.0
    # Defines the primary HSV colour range for the Bonzini playing field.
    green_hsv_range: HSVRange = ((30, 20, 20), (105, 255, 255))
    # Defines a secondary HSV range for cyan or desaturated field surfaces.
    cyan_hsv_range: HSVRange = ((75, 15, 50), (95, 200, 200))
    # Defines the lower-hue HSV range used as red player-colour evidence.
    red_hsv_range: HSVRange = ((0, 80, 40), (15, 255, 255))
    # Defines the upper-hue red range needed because OpenCV HSV hue wraps.
    red_wrap_hsv_range: HSVRange = ((165, 80, 40), (180, 255, 255))
    # Defines the HSV range used as blue player-colour evidence.
    blue_hsv_range: HSVRange = ((90, 40, 30), (140, 255, 255))
    # Sets the square kernel size for field-mask cleanup operations.
    morphology_kernel_size: int = 15
    # Rejects field contours that occupy too little of the source frame.
    minimum_field_area_ratio: float = 0.02
    # Weighs visible field coverage as a secondary confidence signal.
    field_area_confidence_weight: float = 0.25
    # Weighs quadrilateral rectangularity as the primary confidence signal.
    field_rectangularity_confidence_weight: float = 0.75
    # Requires a field candidate to meet this confidence before consensus.
    minimum_field_confidence: float = 0.50
    # Rejects consensus when any corner moves farther than this across frames.
    maximum_field_corner_spread_pixels: float = 40.0
    # Sets the canonical table-plane width used by perspective transforms.
    canonical_field_width: int = 1000
    # Sets the canonical table-plane height used by perspective transforms.
    canonical_field_height: int = 600
    # Sets the lower hysteresis threshold for Canny edge detection.
    canny_low_threshold: int = 30
    # Sets the upper hysteresis threshold for Canny edge detection.
    canny_high_threshold: int = 100
    # Sets the minimum Hough accumulator score for line candidates.
    hough_threshold: int = 40
    # Rejects Hough segments shorter than this number of source pixels.
    hough_min_line_length: int = 50
    # Allows this maximum pixel gap when merging Hough line segments.
    hough_max_line_gap: int = 50
    # Limits accepted rod candidates to near-horizontal line angles.
    maximum_rod_angle_degrees: float = 15.0
    # Requires a rod candidate to span this fraction of field width.
    minimum_rod_length_ratio: float = 0.25
    # Keeps physically distinct rods separated in normalized field coordinates.
    minimum_rod_cluster_separation: float = 0.04
    # Requires a rod cluster to appear in this fraction of accepted frames.
    minimum_rod_coverage_ratio: float = 0.5
    # Enforces the eight-rod geometry of the supported Bonzini layout.
    expected_rod_count: int = 8


DEFAULT_DETECTION_CONFIG = DetectionConfig()
