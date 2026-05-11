from dataclasses import dataclass
from typing import Protocol

import networkx as nx
import numpy as np


class FaceClusterer(Protocol):
    def run(
        self,
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        ...


@dataclass
class SnowballFaceClusterer:
    def run(
        self,
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        n = len(centroids)
        seeds = [int(np.random.randint(n))]
        while len(seeds) < target_k:
            best_idx, max_d = -1, -1.0
            for f_idx in range(n):
                if f_idx in seeds:
                    continue
                min_d = min(
                    float(np.linalg.norm(centroids[f_idx] - centroids[s]))
                    for s in seeds
                )
                if min_d > max_d:
                    max_d, best_idx = min_d, f_idx
            if best_idx == -1:
                break
            seeds.append(best_idx)

        face_to_cluster: dict[int, int] = {s: i for i, s in enumerate(seeds)}
        frontiers: dict[int, list[int]] = {i: [s] for i, s in enumerate(seeds)}
        while len(face_to_cluster) < n:
            progressed = False
            for c_id in range(len(seeds)):
                next_frontier: list[int] = []
                for node in frontiers[c_id]:
                    if node not in dual_base:
                        continue
                    for nbr in dual_base.neighbors(node):
                        if nbr in face_to_cluster:
                            continue
                        face_to_cluster[nbr] = c_id
                        next_frontier.append(nbr)
                        progressed = True
                frontiers[c_id] = next_frontier
            if not progressed:
                break
        return face_to_cluster


@dataclass
class KMeansFaceClusterer:
    max_iter: int = 100
    tolerance: float = 1e-6

    def run(
        self,
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        del dual_base

        points = np.asarray(centroids, dtype=float)
        n = len(points)
        if n == 0:
            return {}

        k = max(1, min(target_k, n))
        center_indices = self._select_initial_centers(points, k)
        centers = points[center_indices].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                points[:, np.newaxis, :] - centers[np.newaxis, :, :],
                axis=2,
            )
            next_labels = np.argmin(distances, axis=1)
            next_centers = centers.copy()

            for cluster_id in range(k):
                members = points[next_labels == cluster_id]
                if len(members) > 0:
                    next_centers[cluster_id] = members.mean(axis=0)

            shift = float(np.linalg.norm(next_centers - centers))
            labels = next_labels
            centers = next_centers
            if shift <= self.tolerance:
                break

        return {face_idx: int(cluster_id) for face_idx, cluster_id in enumerate(labels)}

    @staticmethod
    def _select_initial_centers(points: np.ndarray, k: int) -> list[int]:
        centers = [int(np.random.randint(len(points)))]
        while len(centers) < k:
            best_idx, max_d = -1, -1.0
            for idx in range(len(points)):
                if idx in centers:
                    continue
                min_d = min(
                    float(np.linalg.norm(points[idx] - points[center]))
                    for center in centers
                )
                if min_d > max_d:
                    max_d, best_idx = min_d, idx
            if best_idx == -1:
                break
            centers.append(best_idx)
        return centers


@dataclass
class BalancedFaceGraphClusterer:
    max_iter: int = 10

    def run(
        self,
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        face_count = len(centroids)
        if face_count == 0:
            return {}

        graph = dual_base.copy()
        graph.add_nodes_from(range(face_count))

        target_k = max(1, min(target_k, face_count))
        partitions: list[set[int]] = [
            set(component)
            for component in nx.connected_components(graph)
            if component
        ]

        if not partitions:
            partitions = [set(range(face_count))]

        while len(partitions) < target_k:
            split_index = self._largest_splittable_partition(partitions)
            if split_index is None:
                break

            part = partitions.pop(split_index)
            left, right = self._bisect_partition(graph, part)
            if not left or not right:
                partitions.append(part)
                break

            partitions.extend([left, right])

        return {
            face_idx: cluster_id
            for cluster_id, part in enumerate(partitions)
            for face_idx in part
        }

    @staticmethod
    def _largest_splittable_partition(
        partitions: list[set[int]],
    ) -> int | None:
        candidates = [
            (idx, len(part))
            for idx, part in enumerate(partitions)
            if len(part) > 1
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    def _bisect_partition(
        self,
        graph: nx.Graph,
        partition: set[int],
    ) -> tuple[set[int], set[int]]:
        subgraph = graph.subgraph(partition)
        if subgraph.number_of_nodes() < 2:
            return set(partition), set()

        if subgraph.number_of_edges() == 0:
            return self._fallback_bisection(partition)

        try:
            left, right = nx.community.kernighan_lin_bisection(
                subgraph,
                max_iter=self.max_iter,
            )
            return set(left), set(right)
        except nx.NetworkXError:
            return self._fallback_bisection(partition)

    @staticmethod
    def _fallback_bisection(
        partition: set[int],
    ) -> tuple[set[int], set[int]]:
        ordered = sorted(partition)
        mid = len(ordered) // 2
        return set(ordered[:mid]), set(ordered[mid:])
