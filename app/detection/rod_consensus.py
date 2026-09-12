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
        del field_geometry
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

        rods: list[Rod] = []
        for index, cluster in enumerate(stable_clusters):
            strongest = max(cluster.items, key=lambda item: item[1].confidence)[1]
            rods.append(
                Rod(
                    index=index,
                    line=strongest.line,
                    field_relative_y=float(median(item.field_relative_y for _, item in cluster.items)),
                    confidence=float(median(item.confidence for _, item in cluster.items)),
                    player_colour_evidence=float(median(item.player_colour_evidence for _, item in cluster.items)),
                )
            )
        return rods