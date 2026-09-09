import networkx as nx
import numpy as np

from mr2s_module.cycle import (
    BalancedFaceGraphClusterer,
    FaceClusterPartition,
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
        return dict.fromkeys(range(len(centroids)), 0)


def test_face_cycle_uses_injected_clusterer() -> None:
    clusterer = RecordingClusterer()
    graph = Graph(
        edges=[
            Edge(1, 2, 1, False),
            Edge(2, 3, 1, False),
            Edge(3, 1, 1, False),
        ]
    )

    FaceClusterPartition(clusterer=clusterer).run(graph)

    assert clusterer.calls == 1


def test_face_cycle_boundary_repair_mode_can_remove_only() -> None:
    # boundary/repair 는 도메인 edge id 집합이다 (정점쌍이 아니다).
    boundary_edges = {12, 23}
    repair_edges = {23, 34}

    toggle = FaceClusterPartition(repair_mode="toggle")._apply_boundary_repair(
        boundary_edges,
        repair_edges,
    )
    remove = FaceClusterPartition(repair_mode="remove")._apply_boundary_repair(
        boundary_edges,
        repair_edges,
    )

    assert toggle == {12, 34}
    assert remove == {12}


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
        list(result.values()).count(cluster_id) for cluster_id in set(result.values())
    )

    assert set(result) == set(range(8))
    assert len(set(result.values())) == 4
    assert cluster_sizes == [2, 2, 2, 2]


def test_kmeans_initial_centers_pick_far_apart_points() -> None:
    # 왼쪽에 밀집한 3점 + 멀리 떨어진 2점. farthest-point 초기화라면 어느 점에서
    # 시작하든 떨어진 두 점(인덱스 3, 4)이 반드시 중심으로 뽑힌다.
    points = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [10.0, 0.0], [5.0, 9.0]])

    np.random.seed(0)
    centers = KMeansFaceClusterer._select_initial_centers(points, 3)

    assert len(centers) == len(set(centers)) == 3
    assert {3, 4} <= set(centers)


def test_kmeans_initial_centers_cap_at_point_count() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    np.random.seed(0)
    centers = KMeansFaceClusterer._select_initial_centers(points, 10)

    assert sorted(centers) == [0, 1, 2, 3]


def test_kmeans_initial_centers_handle_duplicate_points() -> None:
    # 중복 좌표가 있어도 같은 인덱스를 두 번 고르지 않는다.
    points = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])

    np.random.seed(0)
    centers = KMeansFaceClusterer._select_initial_centers(points, 4)

    assert sorted(centers) == [0, 1, 2, 3]
