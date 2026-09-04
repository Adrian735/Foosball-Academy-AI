"""Serializable contracts for rod detection and consensus."""

from dataclasses import dataclass
from typing import Any, Tuple

Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]


@dataclass(frozen=True)
class RodCandidate:
    """One per-frame horizontal-line candidate before cross-frame consensus."""

    line: LineSegment
    field_relative_y: float
    length_ratio: float
    player_colour_evidence: float
    confidence: float
    diagnostics: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe candidate details for detector debugging."""
        return {
            "line": [list(point) for point in self.line],
            "field_relative_y": self.field_relative_y,
            "length_ratio": self.length_ratio,
            "player_colour_evidence": self.player_colour_evidence,
            "confidence": self.confidence,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class Rod:
    """A stable physical rod produced by consensus across sampled frames."""

    index: int
    line: LineSegment
    field_relative_y: float
    confidence: float
    player_colour_evidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe stable-rod data for downstream analysis."""
        return {
            "index": self.index,
            "line": [list(point) for point in self.line],
            "field_relative_y": self.field_relative_y,
            "confidence": self.confidence,
            "player_colour_evidence": self.player_colour_evidence,
        }