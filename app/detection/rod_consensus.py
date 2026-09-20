"""Combine per-frame rod candidates into a stable Bonzini rod layout."""

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from app.detection.config import DEFAULT_DETECTION_CONFIG, DetectionConfig
from app.detection.contracts.rod_contracts import Rod, RodCandidate
from app.detection.contracts.table_contracts import FieldGeometry


class RodConsensusError(ValueError):
    """Raised when sampled rod candidates cannot form the expected layout."""


@dataclass
class _RodCluster:
    """Candidates belonging to one physical rod across sampled frames."""

    items: list[tuple[int, RodCandidate]]

    @property
    def last_y(self) -> float:
        """Return the latest normalized position in this sorted cluster."""
        return self.items[-1][1].field_relative_y

    @property
    def coverage(self) -> int:
        """Return the number of distinct frames contributing candidates."""
        return len({frame_index for frame_index, _ in self.items})


class RodConsensus:
    """Cluster normalized rod positions and validate the eight-rod layout."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        """Create a consensus combiner with the shared rod thresholds."""
        self._config = config

    def combine(
        self,
        candidates_by_frame: Iterable[Sequence[RodCandidate]],
        field_geometry: FieldGeometry,
    ) -> list[Rod]:
        """Return stable rods, or raise when coverage does not support eight rods."""
        frames = tuple(tuple(candidates) for candidates in candidates_by_frame)
        if len(frames) < self._config.minimum_accepted_frames:
            raise RodConsensusError("Too few sampled frames for rod consensus")

        items = sorted(
            ((frame_index, candidate) for frame_index, candidates in enumerate(frames) for candidate in candidates),
            key=lambda item: item[1].field_relative_y,
        )
        clusters: list[_RodCluster] = []
        for item in items:
            if not clusters or item[1].field_relative_y - clusters[-1].last_y > self._config.minimum_rod_cluster_separation:
                clusters.append(_RodCluster([item]))
            else:
                clusters[-1].items.append(item)

        minimum_coverage = self._config.minimum_rod_coverage_ratio * len(frames)
        stable_clusters = [cluster for cluster in clusters if cluster.coverage >= minimum_coverage]
        if len(stable_clusters) > self._config.expected_rod_count:
            stable_clusters = sorted(
                stable_clusters,
                key=lambda cluster: (
                    cluster.coverage,
                    max(item[1].confidence for item in cluster.items),
                ),
                reverse=True,
            )[: self._config.expected_rod_count]
            stable_clusters.sort(key=lambda cluster: cluster.items[0][1].field_relative_y)
        if len(stable_clusters) != self._config.expected_rod_count:
            raise RodConsensusError(
                f"Expected {self._config.expected_rod_count} stable rods, found {len(stable_clusters)}"
            )
        self._validate_spacing(stable_clusters)

        rods: list[Rod] = []
        for index, cluster in enumerate(stable_clusters):
            relative_y = float(median(item.field_relative_y for _, item in cluster.items))
            representative = self._representative_candidate(cluster, relative_y)
            rods.append(
                Rod(
                    index=index,
                    line=(
                        representative.line
                        if relative_y < 0.0 or relative_y > 1.0
                        else self._field_width_line(field_geometry, representative.line)
                    ),
                    field_relative_y=relative_y,
                    confidence=float(median(item.confidence for _, item in cluster.items)),
                    player_colour_evidence=float(median(item.player_colour_evidence for _, item in cluster.items)),
                )
            )
        return rods

    def observed_rods(
        self,
        candidates_by_frame: Iterable[Sequence[RodCandidate]],
        field_geometry: FieldGeometry,
    ) -> list[Rod]:
        """Return temporally stable rows for diagnostics without validating the full layout."""
        frames = tuple(tuple(candidates) for candidates in candidates_by_frame)
        if len(frames) < self._config.minimum_accepted_frames:
            return []
        items = sorted(
            ((frame_index, candidate) for frame_index, candidates in enumerate(frames) for candidate in candidates),
            key=lambda item: item[1].field_relative_y,
        )
        clusters: list[_RodCluster] = []
        for item in items:
            if not clusters or item[1].field_relative_y - clusters[-1].last_y > self._config.minimum_rod_cluster_separation:
                clusters.append(_RodCluster([item]))
            else:
                clusters[-1].items.append(item)
        minimum_coverage = self._config.minimum_rod_coverage_ratio * len(frames)
        stable_clusters = [cluster for cluster in clusters if cluster.coverage >= minimum_coverage]
        if len(stable_clusters) > self._config.expected_rod_count:
            stable_clusters = sorted(
                stable_clusters,
                key=lambda cluster: (
                    cluster.coverage,
                    max(item[1].confidence for item in cluster.items),
                ),
                reverse=True,
            )[: self._config.expected_rod_count]
            stable_clusters.sort(key=lambda cluster: cluster.items[0][1].field_relative_y)
        rods: list[Rod] = []
        for index, cluster in enumerate(stable_clusters):
            relative_y = float(median(item.field_relative_y for _, item in cluster.items))
            representative = self._representative_candidate(cluster, relative_y)
            rods.append(
                Rod(
                    index=index,
                    line=(
                        representative.line
                        if relative_y < 0.0 or relative_y > 1.0
                        else self._field_width_line(field_geometry, representative.line)
                    ),
                    field_relative_y=relative_y,
                    confidence=float(median(item.confidence for _, item in cluster.items)),
                    player_colour_evidence=float(median(item.player_colour_evidence for _, item in cluster.items)),
                )
            )
        return rods

    def _validate_spacing(self, clusters: Sequence[_RodCluster]) -> None:
        """Reject layouts with a missing rod or persistent extra row."""
        positions = [median(item.field_relative_y for _, item in cluster.items) for cluster in clusters]
        if not (
            self._config.minimum_top_goal_rod_relative_y
            <= positions[0]
            < self._config.maximum_top_goal_rod_relative_y
        ):
            raise RodConsensusError("Top goalkeeper rod is outside the supported goal-area envelope")
        gaps = [right - left for left, right in zip(positions, positions[1:])]
        expected_gap = (positions[-1] - positions[0]) / (len(positions) - 1)
        if max(abs(gap - expected_gap) for gap in gaps) > self._config.maximum_rod_spacing_deviation:
            raise RodConsensusError("Stable rods do not match the expected Bonzini spacing")

    @staticmethod
    def _representative_candidate(cluster: _RodCluster, relative_y: float) -> RodCandidate:
        """Choose the most central detected fragment, preferring confidence on ties."""
        return min(
            (candidate for _, candidate in cluster.items),
            key=lambda candidate: (abs(candidate.field_relative_y - relative_y), -candidate.confidence),
        )

    @staticmethod
    def _field_width_line(
        field_geometry: FieldGeometry,
        detected_line: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Extend a detected rod line to the perspective field-side intersections."""
        top_left, top_right, bottom_right, bottom_left = field_geometry.corners
        left = RodConsensus._line_intersection(detected_line, (top_left, bottom_left))
        right = RodConsensus._line_intersection(detected_line, (top_right, bottom_right))
        return (left, right) if left is not None and right is not None else detected_line

    @staticmethod
    def _line_intersection(
        first: tuple[tuple[float, float], tuple[float, float]],
        second: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float] | None:
        """Return the intersection of two infinite image lines, if it is finite."""
        (first_start_x, first_start_y), (first_end_x, first_end_y) = first
        (second_start_x, second_start_y), (second_end_x, second_end_y) = second
        first_delta_x = first_end_x - first_start_x
        first_delta_y = first_end_y - first_start_y
        second_delta_x = second_end_x - second_start_x
        second_delta_y = second_end_y - second_start_y
        denominator = first_delta_x * second_delta_y - first_delta_y * second_delta_x
        if abs(denominator) < 1e-6:
            return None
        offset_x = second_start_x - first_start_x
        offset_y = second_start_y - first_start_y
        first_ratio = (offset_x * second_delta_y - offset_y * second_delta_x) / denominator
        return (
            first_start_x + first_ratio * first_delta_x,
            first_start_y + first_ratio * first_delta_y,
        )