"""FlowPolyGenerator 다항식 ↔ 수치 코어(flow_imbalance) 동치성 고정.

flow 균형 점수 `Σ_v (Σ_in w − Σ_out w)²`는 두 형태로 존재한다:
- 수치 코어 `mr2s_module/util/flow_score.py` (Evaluator, SAMR2SSolver 가 위임)
- 기호 다항식 `mr2s_module/qubo/flow_poly_generator.py` (QUBO 계열 솔버)

둘은 직접 공유가 불가능하므로(다항식은 기호식), 이 전수 동치성 테스트가
의미가 갈라지는 것을 막는 접착제 역할을 한다.
"""

import itertools

import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.qubo.flow_poly_generator import FlowPolyGenerator
from mr2s_module.util import flow_imbalance


def _bit_to_direction(edge: Edge, bit: int) -> tuple[int, int]:
    # evaluator._sample_to_directed_edges 와 동일 규약: 0 → (v0, v1), 1 → (v1, v0)
    if bit == 1:
        return (edge.vertices[1], edge.vertices[0])
    return (edge.vertices[0], edge.vertices[1])


def _assert_poly_matches_numeric_core(graph: Graph) -> None:
    poly = FlowPolyGenerator().run(graph)
    variable_edges = [edge for edge in graph.edges.values() if not edge.directed]
    fixed_triples = [
        (edge.vertices[0], edge.vertices[1], float(edge.weight))
        for edge in graph.edges.values()
        if edge.directed
    ]

    for bits in itertools.product((0, 1), repeat=len(variable_edges)):
        sample = {
            edge.to_key(): bit for edge, bit in zip(variable_edges, bits, strict=True)
        }
        triples = fixed_triples + [
            (*_bit_to_direction(edge, bit), float(edge.weight))
            for edge, bit in zip(variable_edges, bits, strict=True)
        ]
        assert poly.energy(sample) == pytest.approx(flow_imbalance(triples))


def test_unweighted_triangle_all_assignments() -> None:
    graph = Graph(
        edges=[
            Edge(0, 1, 1, False),
            Edge(1, 2, 1, False),
            Edge(0, 2, 1, False),
        ]
    )
    _assert_poly_matches_numeric_core(graph)


def test_mixed_weight_triangle_all_assignments() -> None:
    graph = Graph(
        edges=[
            Edge(0, 1, 1, False),
            Edge(1, 2, 2, False),
            Edge(0, 2, 3, False),
        ]
    )
    _assert_poly_matches_numeric_core(graph)


def test_weighted_graph_with_fixed_directed_edge() -> None:
    graph = Graph(
        edges=[
            Edge(1, 0, 4, True),  # 방향 고정, weight 반영 확인
            Edge(0, 2, 2, False),
            Edge(1, 2, 3, False),
            Edge(2, 3, 5, False),
        ]
    )
    _assert_poly_matches_numeric_core(graph)


def test_parallel_edge_multigraph_all_assignments() -> None:
    graph = Graph(
        edges=[
            Edge(0, 1, 3, False),
            Edge(0, 1, 5, False),  # 평행 간선, 독립 변수/독립 weight
            Edge(1, 2, 2, False),
        ]
    )
    _assert_poly_matches_numeric_core(graph)


def test_undirected_weighted_edge_contributes_plus_minus_w() -> None:
    # w≠1 회귀 고정: 한 간선 그래프에서 에너지는 어느 방향이든 2w² (w² + w²).
    weight = 7
    graph = Graph(edges=[Edge(0, 1, weight, False)])
    poly = FlowPolyGenerator().run(graph)
    edge = next(iter(graph.edges.values()))

    for bit in (0, 1):
        assert poly.energy({edge.to_key(): bit}) == pytest.approx(2 * weight**2)
