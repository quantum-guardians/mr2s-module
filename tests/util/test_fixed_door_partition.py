import numpy as np
import pytest

from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.fixed_door_partition import FixedDoorFaceClusterPartition
from tests.util.graph_fixtures import delaunay_graph


def _door_directions(graph: Graph, seed: int, target_k: int) -> dict[int, tuple[int, int]]:
    np.random.seed(seed)
    partition = DoorOnlyFaceClusterPartition(target_k=target_k).run(graph)
    return {
        edge.id: edge.vertices
        for sub_graph in partition.sub_graphs
        for edge in sub_graph.edges.values()
        if edge.directed
    }


@pytest.mark.parametrize("seed", [7, 11])
def test_fixed_directions_survive_partition(seed: int) -> None:
    graph = delaunay_graph(n=60, seed=seed)
    doors = _door_directions(graph, seed, target_k=6)
    assert doors

    # door 를 반대로 박아 넣어도 파티션 결과가 그 방향을 그대로 유지해야 한다.
    flipped = {eid: (head, tail) for eid, (tail, head) in doors.items()}
    for eid, (tail, head) in flipped.items():
        graph.edges[eid].set_direction(tail, head)

    np.random.seed(seed)
    result = FixedDoorFaceClusterPartition(target_k=6).run(graph)

    for sub_graph in result.sub_graphs:
        for edge in sub_graph.edges.values():
            if edge.id in flipped:
                assert edge.directed
                assert edge.vertices == flipped[edge.id]
    # 부모 그래프도 원래 박아둔 방향 그대로여야 한다.
    for eid, direction in flipped.items():
        assert graph.edges[eid].vertices == direction


def test_undirected_input_behaves_like_parent() -> None:
    pairs = [(0, 1), (1, 2), (0, 2)]
    graph = Graph(edges=[Edge(u, v, 1, False) for u, v in pairs])
    result = FixedDoorFaceClusterPartition().run(graph)

    assert len(result.sub_graphs) == 1
    assert all(not e.directed for e in result.sub_graphs[0].edges.values())
