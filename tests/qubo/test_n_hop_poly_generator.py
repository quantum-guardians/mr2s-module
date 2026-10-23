"""NHopPolyGenerator 다항식 ↔ 방향화 후 n-hop 경로 열거의 동치성 고정.

n-hop 항은 "방향이 다 맞아떨어지는 길이 n 짜리 단순 경로"마다 가중치 곱을 보상으로
얹은 것이다. 다항식은 이걸 기호식으로, 아래 참조 구현은 방향을 확정한 뒤 직접
세는 방식으로 계산한다. 둘이 갈라지지 않게 붙잡아 두는 테스트다.

`_get_n_hop_polynomial` 은 내부 누적을 dict 로 하고 마지막에 한 번만
`BinaryPolynomial` 로 옮긴다(ISSUE-99). 그 리팩터가 의미를 바꾸지 않았는지는
호출부에서 보이는 것 — `run()` 이 낸 다항식의 energy — 로만 확인한다.
"""

import itertools
from collections import defaultdict

import pytest
from dimod import Vartype
from dimod.higherorder.polynomial import BinaryPolynomial

from mr2s_module.domain import Edge, Graph
from mr2s_module.qubo import NHop, NHopPolyGenerator, SmallWorldSpec


def _oriented_arcs(
    graph: Graph, sample: dict[str, int]
) -> dict[int, list[tuple[int, float]]]:
    """샘플 비트로 방향을 확정한 tail → [(head, weight)] 인접표.

    비트 규약은 evaluator 와 같다: 0 → (v0, v1), 1 → (v1, v0).
    평행 간선은 서로 독립 arc 로 남는다.
    """
    arcs: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in graph.edges.values():
        tail, head = edge.vertices
        if not edge.directed and sample[edge.to_key()] == 1:
            tail, head = head, tail
        arcs[tail].append((head, float(edge.weight)))
    return arcs


def _path_weight_sum(
    vertex: int,
    remaining: int,
    visited: set[int],
    arcs: dict[int, list[tuple[int, float]]],
) -> float:
    """vertex 에서 출발하는 길이 remaining 단순 유향 경로들의 가중치 곱 합."""
    if remaining == 0:
        return 1.0
    total = 0.0
    for head, weight in arcs[vertex]:
        if head in visited:
            continue
        visited.add(head)
        total += weight * _path_weight_sum(head, remaining - 1, visited, arcs)
        visited.remove(head)
    return total


def _reference_energy(
    graph: Graph, spec: SmallWorldSpec, sample: dict[str, int]
) -> float:
    arcs = _oriented_arcs(graph, sample)
    total = 0.0
    for n_hop in spec.n_hops:
        total += n_hop.weight * sum(
            _path_weight_sum(vertex, n_hop.n, {vertex}, arcs)
            for vertex in graph.get_vertices()
        )
    return -total  # 다항식은 보상을 음수 에너지로 준다


def _assert_matches_path_enumeration(graph: Graph, hops: list[int]) -> None:
    spec = SmallWorldSpec([NHop(n=n, weight=1) for n in hops])
    poly = NHopPolyGenerator(spec).run(graph)
    free_edges = [edge for edge in graph.edges.values() if not edge.directed]

    for bits in itertools.product((0, 1), repeat=len(free_edges)):
        sample = {
            edge.to_key(): bit for edge, bit in zip(free_edges, bits, strict=True)
        }
        assert poly.energy(sample) == pytest.approx(
            _reference_energy(graph, spec, sample)
        )


@pytest.mark.parametrize("hops", [[2], [3], [2, 3]])
def test_matches_path_enumeration_on_weighted_square_with_chord(
    hops: list[int],
) -> None:
    graph = Graph(
        [
            Edge(0, 1, 2, False),
            Edge(1, 2, 3, False),
            Edge(2, 3, 5, False),
            Edge(3, 0, 7, False),
            Edge(0, 2, 11, False),
        ]
    )
    _assert_matches_path_enumeration(graph, hops)


@pytest.mark.parametrize("hops", [[2], [3], [2, 3]])
def test_matches_path_enumeration_with_parallel_edges(hops: list[int]) -> None:
    """평행 간선은 끝점이 같아도 독립 변수라 경로가 따로 세어져야 한다."""
    graph = Graph(
        [
            Edge(0, 1, 2, False),
            Edge(0, 1, 3, False),  # 위와 평행, 가중치 다름
            Edge(1, 2, 5, False),
            Edge(2, 0, 7, False),
            Edge(2, 3, 11, False),
            Edge(3, 0, 13, False),
        ]
    )
    _assert_matches_path_enumeration(graph, hops)


@pytest.mark.parametrize("hops", [[2], [2, 3]])
def test_matches_path_enumeration_with_mixed_directed_edge(hops: list[int]) -> None:
    """이미 방향이 박힌 간선은 변수 없이 상수항으로 들어간다."""
    graph = Graph(
        [
            Edge(0, 1, 2, False),
            Edge(0, 1, 3, True),  # 평행하면서 방향 고정
            Edge(1, 2, 5, False),
            Edge(2, 0, 7, False),
        ]
    )
    _assert_matches_path_enumeration(graph, hops)


def test_parallel_edges_use_independent_variables() -> None:
    graph = Graph(
        [
            Edge(0, 1, 2, False),
            Edge(0, 1, 3, False),
            Edge(0, 1, 5, False),
            Edge(1, 2, 7, False),
            Edge(2, 0, 11, False),
        ]
    )
    poly = NHopPolyGenerator(SmallWorldSpec([NHop(n=2, weight=1)])).run(graph)

    variables = {variable for term in poly for variable in term}
    assert variables == {edge.to_key() for edge in graph.edges.values()}


def test_run_returns_binary_polynomial() -> None:
    """내부 누적은 dict 지만 호출부가 보는 반환형은 BinaryPolynomial 이어야 한다."""
    graph = Graph([Edge(0, 1, 2, False), Edge(1, 2, 3, False), Edge(2, 0, 5, False)])
    poly = NHopPolyGenerator(SmallWorldSpec([NHop(n=2, weight=1)])).run(graph)

    assert isinstance(poly, BinaryPolynomial)
    assert poly.vartype is Vartype.BINARY


def test_empty_graph_returns_empty_polynomial() -> None:
    poly = NHopPolyGenerator(SmallWorldSpec([NHop(n=2, weight=1)])).run(Graph())
    assert len(poly) == 0


def test_missing_spec_raises() -> None:
    graph = Graph([Edge(0, 1, 1, False), Edge(1, 2, 1, False), Edge(2, 0, 1, False)])
    with pytest.raises(ValueError, match="small_world_spec"):
        NHopPolyGenerator().run(graph)
