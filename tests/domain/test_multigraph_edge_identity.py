"""ISSUE-52: 평행 간선(parallel edge) 멀티그래프 지원 검증.

Edge 정체성이 frozenset(endpoints) → 자동 int id 로 전환됐다. 같은 (u,v) 사이 간선
2개가 더 이상 충돌하지 않고 독립 id / 독립 QUBO 변수 / 독립 방향을 갖는지 고정한다.
"""

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.qubo.flow_poly_generator import FlowPolyGenerator
from mr2s_module.util import empty_binary_sample_set


def _empty_solution(edges: dict[int, tuple[int, int]], graph: Graph) -> Solution:
    return Solution(edges=edges, graph=graph, sample_set=empty_binary_sample_set())


def test_edge_ids_are_unique_and_auto_assigned() -> None:
    e1 = Edge(0, 1, 1, False)
    e2 = Edge(0, 1, 1, False)
    assert isinstance(e1.id, int)
    assert e1.id != e2.id  # 같은 끝점이어도 독립 id


def test_oriented_preserves_id_and_sets_direction() -> None:
    e = Edge(0, 1, 7, False)
    oriented = e.oriented(1, 0)
    assert oriented.id == e.id
    assert oriented.directed is True
    assert oriented.vertices == (1, 0)
    assert e.directed is False  # 원본 비파괴
    assert oriented.weight == 7


def test_set_direction_is_in_place_and_keeps_id() -> None:
    e = Edge(0, 1, 1, False)
    original_id = e.id
    e.set_direction(1, 0)
    assert e.id == original_id
    assert e.directed is True
    assert e.vertices == (1, 0)
    assert e.endpoints() == (0, 1)


def test_parallel_edges_keyed_independently_in_graph() -> None:
    e1 = Edge(0, 1, 3, False)
    e2 = Edge(0, 1, 5, False)
    graph = Graph(edges=[e1, e2])

    assert len(graph.edges) == 2  # 안 뭉개짐
    assert graph.edges[e1.id] is e1
    assert graph.edges[e2.id] is e2
    assert {e.weight for e in graph.edges.values()} == {3, 5}


def test_parallel_edges_produce_distinct_qubo_variables() -> None:
    graph = Graph(edges=[Edge(0, 1, 3, False), Edge(0, 1, 5, False)])
    poly = FlowPolyGenerator().run(graph)

    variables = {v for term in poly for v in term}
    expected = {edge.to_key() for edge in graph.edges.values()}
    assert len(expected) == 2
    assert expected.issubset(variables)  # 평행 간선마다 독립 변수


def test_eval_flow_sums_parallel_edge_weights_without_collapsing() -> None:
    e1 = Edge(0, 1, 3, True)
    e2 = Edge(0, 1, 5, True)
    graph = Graph(edges=[e1, e2])
    # 두 평행 간선 모두 0 → 1 방향.
    solution = _empty_solution({e1.id: (0, 1), e2.id: (0, 1)}, graph)

    # out(0)=3+5=8, in(1)=8 → 64 + 64. 뭉갰다면 한쪽 weight 만 반영돼 값이 달라진다.
    assert Evaluator().eval_flow(solution) == 128.0


def test_solution_edges_roundtrip_is_id_keyed() -> None:
    e1 = Edge(0, 1, 1, False)
    e2 = Edge(0, 1, 1, False)
    graph = Graph(edges=[e1, e2])
    solution = _empty_solution({e1.id: (0, 1), e2.id: (1, 0)}, graph)

    assert solution.edges[e1.id] == (0, 1)
    assert solution.edges[e2.id] == (1, 0)  # 평행 간선이 반대 방향 독립 보존
