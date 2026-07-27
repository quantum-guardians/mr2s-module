import numpy as np
import pytest

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.graph_fixtures import delaunay_graph


def graph_from_pairs(pairs: list[tuple[int, int]]) -> Graph:
    return Graph(edges=[Edge(u, v, 1, False) for u, v in pairs])


def _subgraph_owner_counts(result) -> dict[int, list[tuple[int, bool]]]:
    """edge id → [(subgraph_idx, directed), ...]"""
    owners: dict[int, list[tuple[int, bool]]] = {}
    for sg_idx, sg in enumerate(result.sub_graphs):
        for edge in sg.edges.values():
            owners.setdefault(edge.id, []).append((sg_idx, edge.directed))
    return owners


def test_single_macro_triangle_emits_no_directed_edges() -> None:
    # Triangle: macro 1개 → door 없음 → 전 간선 undirected free.
    graph = graph_from_pairs([(0, 1), (1, 2), (0, 2)])
    result = DoorOnlyFaceClusterPartition().run(graph)

    assert len(result.sub_graphs) == 1
    assert all(not e.directed for e in result.sub_graphs[0].edges.values())
    assert {e.id for e in result.sub_graphs[0].edges.values()} == set(
        graph.edges.keys()
    )


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_door_only_directs_only_shared_boundary(seed: int) -> None:
    graph = delaunay_graph(n=60, seed=seed)

    # 같은 np.random 시드 → base/variant 동일 클러스터링 → macro 구조 비교 가능.
    np.random.seed(seed)
    base = FaceClusterPartition(target_k=6).run(graph)
    np.random.seed(seed)
    variant = DoorOnlyFaceClusterPartition(target_k=6).run(graph)

    base_directed = {e.id for e in base.directed_edges()}
    variant_directed = {e.id for e in variant.directed_edges()}

    # door 는 base 외곽선의 부분집합이고, 진부분집합이어야 함 (outer 가 풀림).
    assert variant_directed <= base_directed
    assert variant_directed < base_directed

    owners = _subgraph_owner_counts(variant)
    for edge_id, owner_list in owners.items():
        if edge_id in variant_directed:
            # door: 인접 macro 양쪽에 directed 로 중복 삽입.
            assert len(owner_list) == 2
            assert all(directed for _, directed in owner_list)
        else:
            # outer/intra/내부: 단일 macro 에 undirected.
            assert len(owner_list) == 1
            assert not owner_list[0][1]


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_door_only_partition_is_complete_cover(seed: int) -> None:
    graph = delaunay_graph(n=60, seed=seed)
    np.random.seed(seed)
    result = DoorOnlyFaceClusterPartition(target_k=6).run(graph)

    input_ids = set(graph.edges.keys())
    covered_ids = {e.id for sg in result.sub_graphs for e in sg.edges.values()} | {
        e.id for e in result.remaining_edges
    }
    assert covered_ids == input_ids
