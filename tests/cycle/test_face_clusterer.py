import networkx as nx
import numpy as np

from mr2s_module.cycle import (
    BalancedFaceGraphClusterer,
    FaceCycle,
    KMeansFaceClusterer,
)
from mr2s_module.domain import Edge, Graph


class RecordingClusterer:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        self.calls += 1
        return {face_idx: 0 for face_idx in range(len(centroids))}


def test_face_cycle_uses_injected_clusterer() -> None:
    clusterer = RecordingClusterer()
    graph = Graph(edges=[
        Edge(1, 2, 1, False),
        Edge(2, 3, 1, False),
        Edge(3, 1, 1, False),
    ])

    FaceCycle(clusterer=clusterer).run(graph)

    assert clusterer.calls == 1


def test_face_cycle_boundary_repair_mode_can_remove_only() -> None:
    boundary_edges = {(1, 2), (2, 3)}
    repair_edges = {(2, 3), (3, 4)}

    toggle = FaceCycle(repair_mode="toggle")._apply_boundary_repair(
        boundary_edges,
        repair_edges,
    )
    remove = FaceCycle(repair_mode="remove")._apply_boundary_repair(
        boundary_edges,
        repair_edges,
    )

    assert toggle == {(1, 2), (3, 4)}
    assert remove == {(1, 2)}


def test_kmeans_face_clusterer_assigns_each_face_to_cluster() -> None:
    centroids = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.1]),
        np.array([10.0, 10.0]),
        np.array([10.0, 10.1]),
    ]

    np.random.seed(0)
    result = KMeansFaceClusterer().run(
        centroids=centroids,
        dual_base=nx.Graph(),
        target_k=2,
    )

    assert set(result) == {0, 1, 2, 3}
    assert len(set(result.values())) == 2


def test_balanced_face_graph_clusterer_splits_dual_graph() -> None:
    dual_base = nx.path_graph(8)
    centroids = [np.array([float(idx), 0.0]) for idx in range(8)]

    result = BalancedFaceGraphClusterer().run(
        centroids=centroids,
        dual_base=dual_base,
        target_k=4,
    )
    cluster_sizes = sorted(
        list(result.values()).count(cluster_id)
        for cluster_id in set(result.values())
    )

    assert set(result) == set(range(8))
    assert len(set(result.values())) == 4
    assert cluster_sizes == [2, 2, 2, 2]
