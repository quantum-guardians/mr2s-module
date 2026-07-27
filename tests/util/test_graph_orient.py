import networkx as nx

from mr2s_module.domain import Edge, Graph
from mr2s_module.util import (
    domain_graph_to_networkx_multi,
    flow_imbalance,
    robbins_orient,
)


def _pair_imbalance(oriented: dict[int, Edge]) -> float:
    return flow_imbalance(
        (*edge.vertices, float(edge.weight)) for edge in oriented.values()
    )


def test_robbins_orient_keys_result_by_domain_edge_id() -> None:
    edge_a = Edge(0, 1, 1, False)
    edge_b = Edge(1, 2, 1, False)
    edge_c = Edge(0, 2, 1, False)
    graph = Graph(edges=[edge_a, edge_b, edge_c])

    oriented = robbins_orient(domain_graph_to_networkx_multi(graph), 0)

    assert set(oriented.keys()) == {edge_a.id, edge_b.id, edge_c.id}
    for edge_id, edge in oriented.items():
        assert edge.id == edge_id
        assert edge.directed

    D = nx.DiGraph()
    D.add_edges_from(edge.vertices for edge in oriented.values())
    assert nx.is_strongly_connected(D)


def test_robbins_orient_balances_parallel_copies_by_weight() -> None:
    # weights {1, 2, 3}: 완전 상쇄 파티션 {3} vs {1, 2} 이 존재 → net 0.
    # weight 무시 교대(index%2)는 이 파티션을 놓쳐 불필요한 불균형을 남긴다.
    edges = [Edge(0, 1, 1, False), Edge(0, 1, 2, False), Edge(0, 1, 3, False)]
    graph = Graph(edges=edges)

    oriented = robbins_orient(domain_graph_to_networkx_multi(graph), 0)

    assert set(oriented.keys()) == {e.id for e in edges}
    # 이 쌍만 있으므로 flow_imbalance 가 곧 쌍 기여. weight-aware 배정은 0.
    assert _pair_imbalance(oriented) == 0.0

    D = nx.DiGraph()
    D.add_edges_from(edge.vertices for edge in oriented.values())
    assert nx.is_strongly_connected(D)


def test_robbins_orient_alternates_parallel_copies() -> None:
    edge_a = Edge(0, 1, 3, False)
    edge_b = Edge(0, 1, 5, False)
    graph = Graph(edges=[edge_a, edge_b])

    oriented = robbins_orient(domain_graph_to_networkx_multi(graph), 0)

    assert set(oriented.keys()) == {edge_a.id, edge_b.id}
    assert oriented[edge_a.id].vertices == tuple(reversed(oriented[edge_b.id].vertices))
    assert oriented[edge_a.id].weight == 3
    assert oriented[edge_b.id].weight == 5

    D = nx.DiGraph()
    D.add_edges_from(edge.vertices for edge in oriented.values())
    assert nx.is_strongly_connected(D)
