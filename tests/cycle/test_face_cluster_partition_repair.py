"""경계 봉합(T-join) 옵션: repair_terminals / boundary_weight (ISSUE-98)."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from mr2s_module.cycle import FaceClusterPartition, KMeansFaceClusterer
from mr2s_module.domain import Graph
from tests.util.graph_fixtures import delaunay_graph, graph_from_pairs


@dataclass
class DualPathClusterer:
    """쌍대 그래프가 경로일 때, 한쪽 끝 leaf 와 그 이웃을 군집 0, 나머지를 1 로."""

    def run(self, centroids, dual_base, target_k):
        del centroids, target_k
        leaf = min(node for node in dual_base.nodes if dual_base.degree(node) == 1)
        neighbor = next(iter(dual_base.neighbors(leaf)))
        return {
            node: (0 if node in {leaf, neighbor} else 1) for node in dual_base.nodes
        }


@dataclass
class DualCycleThirdsClusterer:
    """쌍대 그래프가 6-사이클일 때 연속한 두 면씩 군집 0/1/2 — 중심 정점이 내부 Y자가 된다."""

    def run(self, centroids, dual_base, target_k):
        del centroids, target_k
        start = min(dual_base.nodes)
        order = [start]
        prev = None
        while len(order) < dual_base.number_of_nodes():
            nxt = next(n for n in dual_base.neighbors(order[-1]) if n != prev)
            prev, order = order[-1], [*order, nxt]
        return {face: index // 2 for index, face in enumerate(order)}


def _strip_graph() -> Graph:
    """정사각형 4개가 한 줄로 붙은 띠: 정점 (i, j) → id i + 5 * j."""
    pairs = []
    for j in range(2):
        for i in range(4):
            pairs.append((i + 5 * j, i + 1 + 5 * j))
    for i in range(5):
        pairs.append((i, i + 5))
    return graph_from_pairs(pairs)


def _wheel_graph() -> Graph:
    """중심 0 과 테두리 1..6 의 바퀴: 삼각형 면 6개."""
    pairs = [(0, r) for r in range(1, 7)] + [(r, r % 6 + 1) for r in range(1, 7)]
    return graph_from_pairs(pairs)


def _directed_ids(result) -> set[int]:
    return {e.id for e in result.directed_edges()}


def _free_ids(result) -> set[int]:
    return {
        e.id for sg in result.sub_graphs for e in sg.edges.values() if not e.directed
    }


def _is_consistently_oriented(result) -> bool:
    seen: dict[int, tuple[int, int]] = {}
    for sg in result.sub_graphs:
        for e in sg.edges.values():
            if e.directed and seen.setdefault(e.id, e.endpoints()) != e.endpoints():
                return False
    return True


def test_invalid_repair_terminals_raises() -> None:
    with pytest.raises(ValueError):
        FaceClusterPartition(repair_terminals="hull")


def test_negative_boundary_weight_raises() -> None:
    with pytest.raises(ValueError):
        FaceClusterPartition(boundary_weight=-1)


def test_default_options_match_explicit_legacy_options() -> None:
    graph = delaunay_graph(n=60, seed=7)
    np.random.seed(7)
    default = FaceClusterPartition(target_k=6, clusterer=KMeansFaceClusterer()).run(
        graph
    )
    np.random.seed(7)
    explicit = FaceClusterPartition(
        target_k=6,
        clusterer=KMeansFaceClusterer(),
        repair_terminals="all",
        boundary_weight=1,
    ).run(graph)
    assert [{e.id for e in sg.edges.values()} for sg in default.sub_graphs] == [
        {e.id for e in sg.edges.values()} for sg in explicit.sub_graphs
    ]
    assert _directed_ids(default) == _directed_ids(explicit)


@pytest.mark.parametrize("seed", [0, 7, 11, 23])
def test_interior_merge_partition_is_complete_cover(seed: int) -> None:
    graph = delaunay_graph(n=60, seed=seed)
    np.random.seed(seed)
    result = FaceClusterPartition(
        target_k=6,
        clusterer=KMeansFaceClusterer(),
        repair_terminals="interior",
        boundary_weight=0,
    ).run(graph)

    input_ids = set(graph.edges.keys())
    assert _directed_ids(result).issubset(input_ids)
    assert len(result.sub_graphs) >= 1
    covered = {e.id for sg in result.sub_graphs for e in sg.edges.values()} | {
        e.id for e in result.remaining_edges
    }
    assert covered == input_ids
    assert _is_consistently_oriented(result)


def test_interior_terminals_leave_hull_odd_vertices_alone() -> None:
    # 띠의 가운데 세로 간선 하나만 군집 경계다. 그 양 끝은 외벽 위 홀수(차수 3) 정점.
    # 현행은 둘을 짝지어 그 간선 자체를 XOR 로 지워 거대면 1개가 되지만,
    # interior 모드는 외벽 홀수를 건드리지 않아 군집 2개가 그대로 거대면이 된다.
    graph = _strip_graph()
    legacy = FaceClusterPartition(target_k=2, clusterer=DualPathClusterer()).run(graph)
    interior = FaceClusterPartition(
        target_k=2, clusterer=DualPathClusterer(), repair_terminals="interior"
    ).run(graph)

    assert len(legacy.sub_graphs) == 1
    assert len(interior.sub_graphs) == 2
    # 외벽 10개(가로 8 + 양끝 세로 2) + 가운데 세로 간선 1개가 고정,
    # 나머지 세로 간선 2개가 자유 변수 (간선 13개 전부 계수됨)
    assert len(_directed_ids(interior)) == 11
    assert len(_free_ids(interior)) == 2
    assert _is_consistently_oriented(interior)


def test_zero_boundary_weight_erases_a_dangling_boundary_at_interior_y() -> None:
    # 바퀴 중심에서 군집 세 개가 만난다(내부 Y자, 경계 차수 3). 경계 비용 0 이면
    # 수리 경로가 중심에서 바퀴살 경계 하나를 따라 외벽(접지)으로 나가며 그 살을 지운다.
    graph = _wheel_graph()
    result = FaceClusterPartition(
        target_k=3,
        clusterer=DualCycleThirdsClusterer(),
        repair_terminals="interior",
        boundary_weight=0,
    ).run(graph)

    assert len(result.sub_graphs) == 2
    # 테두리 6개 + 남은 경계 바퀴살 2개가 고정, 바퀴살 4개가 자유 변수
    assert len(_directed_ids(result)) == 8
    assert len(_free_ids(result)) == 4
    assert _is_consistently_oriented(result)
    merged_dual = nx.Graph()
    for i, a in enumerate(result.sub_graphs):
        for j, b in enumerate(result.sub_graphs):
            if i < j and _directed_ids_of(a) & _directed_ids_of(b):
                merged_dual.add_edge(i, j)
    assert nx.is_bipartite(merged_dual)


def _directed_ids_of(sub_graph) -> set[int]:
    return {e.id for e in sub_graph.edges.values() if e.directed}


def test_odd_multiplicity_edges_drops_shared_path_edges() -> None:
    # 두 경로가 간선 7 을 공유하면 T-join(대칭차)에서는 빠져야 한다.
    path_a = [1, ("edge", 5), 2, ("edge", 7), 3]
    path_b = [4, ("edge", 9), 2, ("edge", 7), 3]
    assert FaceClusterPartition._odd_multiplicity_edges([path_a, path_b]) == {5, 9}
