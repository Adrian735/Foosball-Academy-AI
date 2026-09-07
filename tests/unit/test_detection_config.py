"""Tests for detector configuration defaults and overrides."""

from dataclasses import FrozenInstanceError, replace

import pytest

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig


def test_detection_config_has_bonzini_defaults() -> None:
    """The default configuration encodes the initial supported table layout."""
    config = DetectionConfig()

    assert config.expected_rod_count == 8
    assert config.minimum_video_width == 1280
    assert config.minimum_video_height == 720
    assert config.minimum_video_duration_seconds == 5.0
    assert config.maximum_field_corner_spread_pixels == 40.0


def test_detection_config_can_be_overridden_without_global_mutation() -> None:
    """A feature/test can override thresholds by creating a separate config."""
    custom_config = replace(DEFAULT_DETECTION_CONFIG, minimum_field_confidence=0.8)

    assert custom_config.minimum_field_confidence == 0.8
    assert DEFAULT_DETECTION_CONFIG.minimum_field_confidence == 0.50


def test_detection_config_is_immutable() -> None:
    """Detector configuration cannot be changed accidentally during analysis."""
    with pytest.raises(FrozenInstanceError):
        DEFAULT_DETECTION_CONFIG.expected_rod_count = 7  # type: ignore[misc]
