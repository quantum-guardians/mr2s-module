"""ISSUE-52: 체인 축약 + lift."""

import networkx as nx
import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.reduction import (
    SuperEdgeWeight,
    contract_chains,
    lift_solution_edges,
)


def _k4_edges() -> list[Edge]:
    return [
        Edge(0, 1, 1, False),
        Edge(0, 2, 1, False),
        Edge(0, 3, 1, False),
        Edge(1, 2, 1, False),
        Edge(1, 3, 1, False),
        Edge(2, 3, 1, False),
    ]


def _k4_with_chain(chain_length: int) -> tuple[Graph, list[Edge]]:
    body = _k4_edges()
    chain = []
    prev = 0
    for i in range(chain_length - 1):
        chain.append(Edge(prev, 10 + i, 1, False))
        prev = 10 + i
    chain.append(Edge(prev, 1, 1, False))
    return Graph(edges=body + chain), chain


def _all_endpoint_directions(graph: Graph) -> dict[int, tuple[int, int]]:
    return {edge_id: edge.endpoints() for edge_id, edge in graph.edges.items()}


def test_contract_replaces_chain_with_super_edge() -> None:
    graph, chain_edges = _k4_with_chain(4)
    result = contract_chains(graph)

    assert len(result.chain_by_super_id) == 1
    assert not result.cycle_chains
    assert len(result.contracted_graph.edges) == len(graph.edges) - 4 + 1
    super_id = next(iter(result.chain_by_super_id))
    super_edge = result.contracted_graph.edges[super_id]
    assert super_edge.weight == 1  # harmonic 1/4 → 1 클램프
    assert super_edge.directed is False
    assert set(super_edge.endpoints()) == {0, 1}
    # 체인 원본 간선은 축약 그래프에서 제거됨
    for edge in chain_edges:
        assert edge.id not in result.contracted_graph.edges
    # 원본 그래프 비변형
    assert len(graph.edges) == 10


def test_contract_clones_pass_through_edges() -> None:
    graph, _ = _k4_with_chain(3)
    result = contract_chains(graph)

    for edge_id, clone in result.contracted_graph.edges.items():
        if edge_id in result.chain_by_super_id:
            continue
        original = graph.edges[edge_id]
        assert clone is not original  # in-place set_direction 오염 방지
        assert clone.id == original.id
        assert clone.endpoints() == original.endpoints()
        assert clone.weight == original.weight


def test_contract_clones_even_without_chains() -> None:
    graph = Graph(edges=_k4_edges())
    result = contract_chains(graph)

    assert not result.has_chains
    assert set(result.contracted_graph.edges) == set(graph.edges)
    for edge_id, clone in result.contracted_graph.edges.items():
        assert clone is not graph.edges[edge_id]


@pytest.mark.parametrize("forward", [True, False])
def test_lift_expands_super_edge_uniformly(forward: bool) -> None:
    graph, chain_edges = _k4_with_chain(4)
    result = contract_chains(graph)
    super_id, chain = next(iter(result.chain_by_super_id.items()))
    a, b = chain.endpoints

    solved = {
        edge_id: result.contracted_graph.edges[edge_id].endpoints()
        for edge_id in result.contracted_graph.edges
        if edge_id != super_id
    }
    solved[super_id] = (a, b) if forward else (b, a)

    lifted = lift_solution_edges(solved, result)

    # 원본 간선 전부 방향 부여, super id 는 소멸
    assert set(lifted) == set(graph.edges)
    # 체인 내부 정점은 in 1 / out 1 (균일 방향)
    dg = nx.DiGraph()
    dg.add_edges_from(lifted[e.id] for e in chain_edges)
    for vertex in chain.interior_vertices:
        assert dg.in_degree(vertex) == 1
        assert dg.out_degree(vertex) == 1
    # 방향이 super edge 방향과 일치
    source = a if forward else b
    sink = b if forward else a
    assert dg.out_degree(source) == 1 and dg.in_degree(source) == 0
    assert dg.in_degree(sink) == 1 and dg.out_degree(sink) == 0


def test_lift_rejects_direction_not_matching_endpoints() -> None:
    graph, _ = _k4_with_chain(3)
    result = contract_chains(graph)
    super_id = next(iter(result.chain_by_super_id))

    with pytest.raises(ValueError):
        lift_solution_edges({super_id: (98, 99)}, result)


def test_lift_rejects_missing_super_edge() -> None:
    graph, _ = _k4_with_chain(3)
    result = contract_chains(graph)
    super_id = next(iter(result.chain_by_super_id))

    solved = {
        edge_id: result.contracted_graph.edges[edge_id].endpoints()
        for edge_id in result.contracted_graph.edges
        if edge_id != super_id
    }
    with pytest.raises(ValueError, match="missing super edge"):
        lift_solution_edges(solved, result)


def test_harmonic_super_edge_weight_preserves_distance() -> None:
    # 체인 가중치 (10, 10): 거리 = 1/10 + 1/10 = 0.2 → W = 5
    chain = [Edge(0, 10, 10, False), Edge(10, 1, 10, False)]
    result = contract_chains(Graph(edges=_k4_edges() + chain))
    super_edge = result.contracted_graph.edges[next(iter(result.chain_by_super_id))]
    assert super_edge.weight == pytest.approx(5.0)


def test_harmonic_super_edge_weight_clamped_to_one() -> None:
    # 단위 가중치 체인 3간선: W = 1/3 → 1 로 강제
    chain = [
        Edge(0, 10, 1, False),
        Edge(10, 11, 1, False),
        Edge(11, 1, 1, False),
    ]
    result = contract_chains(Graph(edges=_k4_edges() + chain))
    super_edge = result.contracted_graph.edges[next(iter(result.chain_by_super_id))]
    assert super_edge.weight == 1.0


def test_unit_super_edge_weight_mode() -> None:
    chain = [Edge(0, 10, 10, False), Edge(10, 1, 10, False)]
    result = contract_chains(
        Graph(edges=_k4_edges() + chain), super_edge_weight=SuperEdgeWeight.UNIT
    )
    super_edge = result.contracted_graph.edges[next(iter(result.chain_by_super_id))]
    assert super_edge.weight == 1


def test_hanging_cycle_removed_and_lifted_uniformly() -> None:
    # K4 의 정점 0 에 매달린 사이클 0-20-21-0: self-loop super edge 대신 제거.
    cycle = [Edge(0, 20, 1, False), Edge(20, 21, 1, False), Edge(21, 0, 1, False)]
    graph = Graph(edges=_k4_edges() + cycle)
    result = contract_chains(graph)

    assert len(result.cycle_chains) == 1
    assert not result.chain_by_super_id
    # 사이클 간선·내부 정점이 축약 그래프에서 사라짐 (QUBO 변수 0개)
    for edge in cycle:
        assert edge.id not in result.contracted_graph.edges
    assert {20, 21} & result.contracted_graph.get_vertices() == set()

    lifted = lift_solution_edges(
        _all_endpoint_directions(result.contracted_graph), result
    )
    assert set(lifted) == set(graph.edges)
    # 사이클은 균일 회전: 각 사이클 정점 in 1 / out 1, 부착점 경유 왕복 가능
    dg = nx.DiGraph()
    dg.add_edges_from(lifted[e.id] for e in cycle)
    for vertex in (0, 20, 21):
        assert dg.in_degree(vertex) == 1
        assert dg.out_degree(vertex) == 1
    assert nx.is_strongly_connected(dg)


def test_whole_graph_cycle_contracts_to_empty_graph() -> None:
    triangle = [Edge(0, 1, 1, False), Edge(1, 2, 1, False), Edge(2, 0, 1, False)]
    graph = Graph(edges=triangle)
    result = contract_chains(graph)

    assert result.contracted_graph.is_empty()
    assert len(result.cycle_chains) == 1

    lifted = lift_solution_edges({}, result)
    assert set(lifted) == set(graph.edges)
    dg = nx.DiGraph(lifted.values())
    assert nx.is_strongly_connected(dg)


def test_cycle_removal_exposes_new_chain_fixed_point() -> None:
    # h 는 K4 정점. 삼각형 h-a-b + a 의 매달린 사이클 a-c1-c2-a.
    # 1라운드: b 축약(super1: h-a), 사이클 a-c1-c2-a 제거 → a 차수 2 로 하락.
    # 2라운드: a 가 새 내부 정점 → h-a 간선과 super1 이 매달린 사이클(h==h)로 제거.
    # lift 는 사이클 안의 super1 을 중첩 전개해야 한다.
    h = 0
    extra = [
        Edge(h, 30, 1, False),  # h-a
        Edge(30, 31, 1, False),  # a-b
        Edge(31, h, 1, False),  # b-h
        Edge(30, 32, 1, False),  # a-c1
        Edge(32, 33, 1, False),  # c1-c2
        Edge(33, 30, 1, False),  # c2-a
    ]
    graph = Graph(edges=_k4_edges() + extra)
    result = contract_chains(graph)

    # 고정점 도달: K4 만 남는다
    assert set(result.contracted_graph.edges) == {e.id for e in _all_k4_ids(graph)}
    assert len(result.cycle_chains) == 2

    lifted = lift_solution_edges(
        _all_endpoint_directions(result.contracted_graph), result
    )
    assert set(lifted) == set(graph.edges)
    # 제거된 6개 간선만으로 h·a·b·c1·c2 가 서로 도달 가능해야 한다
    dg = nx.DiGraph()
    dg.add_edges_from(lifted[e.id] for e in extra)
    assert nx.is_strongly_connected(dg)


def _all_k4_ids(graph: Graph) -> list[Edge]:
    return [
        edge for edge in graph.edges.values() if set(edge.endpoints()) <= {0, 1, 2, 3}
    ]


def test_parallel_super_edges_between_same_endpoints() -> None:
    # 두 허브 사이 체인 2개 + 직결 간선 1개 (theta graph). 허브 차수는 K4 로 보강.
    h1, h2 = 0, 1
    chains = [
        Edge(h1, 40, 1, False),
        Edge(40, h2, 1, False),  # 체인 1
        Edge(h1, 41, 1, False),
        Edge(41, h2, 1, False),  # 체인 2
    ]
    graph = Graph(edges=_k4_edges() + chains)
    result = contract_chains(graph)

    assert len(result.chain_by_super_id) == 2
    # 축약 그래프에 h1-h2 평행 간선 3개: 직결 1 + super 2 (멀티그래프 유지)
    parallel = [
        edge
        for edge in result.contracted_graph.edges.values()
        if set(edge.endpoints()) == {h1, h2}
    ]
    assert len(parallel) == 3
