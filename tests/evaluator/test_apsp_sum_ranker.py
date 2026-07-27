from typing import cast

import pytest

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.evaluator.apsp_sum_ranker import ApspMethod
from mr2s_module.util import empty_binary_sample_set


def _build_solution(
    edges: list[Edge], directed_edges: set[tuple[int, int]]
) -> Solution:
    graph = Graph(edges=edges)
    edges_by_id: dict[int, tuple[int, int]] = {}
    for edge in graph.edges.values():
        if edge.directed:
            edges_by_id[edge.id] = edge.vertices
            continue
        u, v = edge.endpoints()
        if (u, v) in directed_edges:
            edges_by_id[edge.id] = (u, v)
        elif (v, u) in directed_edges:
            edges_by_id[edge.id] = (v, u)
    return Solution(
        edges=edges_by_id,
        graph=graph,
        sample_set=empty_binary_sample_set(),
    )


def _triangle_cycle(weight: int = 1) -> Solution:
    return _build_solution(
        edges=[
            Edge(1, 2, weight, False),
            Edge(2, 3, weight, False),
            Edge(1, 3, weight, False),
        ],
        directed_edges={(1, 2), (2, 3), (3, 1)},
    )


def _digon() -> Solution:
    # 서로 반대 방향 평행 간선 → 모든 쌍 거리 보존(완벽 방향화)
    return _build_solution(
        edges=[Edge(1, 2, 1, True), Edge(2, 1, 1, True)],
        directed_edges=set(),
    )


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        # 존재하지 않는 method 를 일부러 넣는다 — cast 로 타입 게이트를 통과시켜 런타임 검증.
        ApspSumRanker(method=cast(ApspMethod, "hop"))


# --- method="stretch" (기본) ---


def test_default_method_is_stretch() -> None:
    assert ApspSumRanker().method == "stretch"


def test_stretch_of_triangle_cycle_is_1_5() -> None:
    # 정방향 3쌍 stretch=1, 역방향 3쌍 stretch=2 → 평균 1.5
    assert ApspSumRanker().run(_triangle_cycle()) == pytest.approx(1.5)


def test_stretch_is_invariant_to_weight_scale() -> None:
    assert ApspSumRanker().run(_triangle_cycle(weight=10)) == pytest.approx(1.5)


def test_stretch_returns_inf_when_not_strongly_connected() -> None:
    solution = _build_solution(
        edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, False)],
        directed_edges={(1, 2), (2, 3)},
    )

    assert ApspSumRanker().run(solution) == float("inf")


def test_stretch_is_one_for_distance_preserving_orientation() -> None:
    assert ApspSumRanker().run(_digon()) == pytest.approx(1.0)


def test_stretch_reuses_undirected_cache_for_same_graph() -> None:
    ranker = ApspSumRanker()
    solution = _triangle_cycle()

    first = ranker.run(solution)
    second = ranker.run(solution)

    assert first == second == pytest.approx(1.5)
    assert len(ranker._undirected_cache._cache) == 1


# --- method="efficiency" (1 - E_dir/E_und, 낮을수록 좋음) ---


def test_efficiency_of_triangle_cycle_is_0_25() -> None:
    # E_dir = 3·1 + 3·(1/2) = 4.5, E_und = 6 → 1 - 0.75 = 0.25
    ranker = ApspSumRanker(method="efficiency")
    assert ranker.run(_triangle_cycle()) == pytest.approx(0.25)


def test_efficiency_gives_partial_score_when_not_strongly_connected() -> None:
    # 일렬 1→2→3→4: 정방향 6쌍만 도달(거리 보존), 역방향 6쌍 기여 0 → 1 - 0.5
    solution = _build_solution(
        edges=[
            Edge(1, 2, 1, False),
            Edge(2, 3, 1, False),
            Edge(3, 4, 1, False),
        ],
        directed_edges={(1, 2), (2, 3), (3, 4)},
    )

    assert ApspSumRanker(method="efficiency").run(solution) == pytest.approx(0.5)


def test_efficiency_is_zero_for_distance_preserving_orientation() -> None:
    assert ApspSumRanker(method="efficiency").run(_digon()) == pytest.approx(0.0)


def test_efficiency_ranks_less_disconnected_solution_lower() -> None:
    # 같은 4-사이클 그래프: 완전 사이클 방향 vs 한 간선 역방향(끊김)
    def four_cycle(directed_edges: set[tuple[int, int]]) -> Solution:
        return _build_solution(
            edges=[
                Edge(1, 2, 1, False),
                Edge(2, 3, 1, False),
                Edge(3, 4, 1, False),
                Edge(1, 4, 1, False),
            ],
            directed_edges=directed_edges,
        )

    strong = four_cycle({(1, 2), (2, 3), (3, 4), (4, 1)})
    broken = four_cycle({(1, 2), (2, 3), (3, 4), (1, 4)})

    ranker = ApspSumRanker(method="efficiency")
    assert ranker.run(strong) < ranker.run(broken)


# --- method="sum" (거리 = 1/weight 인 APSP 합) ---


def test_sum_uses_inverse_weight_as_distance() -> None:
    # 방향 사이클 1→2→3→1, weight 1/2/4 → 거리 1/0.5/0.25.
    # 1→2:1, 1→3:1.5, 2→3:0.5, 2→1:0.75, 3→1:0.25, 3→2:1.25 = 5.25
    solution = _build_solution(
        edges=[
            Edge(1, 2, 1, False),
            Edge(2, 3, 2, False),
            Edge(1, 3, 4, False),
        ],
        directed_edges={(1, 2), (2, 3), (3, 1)},
    )

    assert ApspSumRanker(method="sum").run(solution) == pytest.approx(5.25)


def test_sum_prefers_heavy_detour_over_light_direct_edge() -> None:
    # 1→3 직행(w=1, 거리 1)보다 1→2→3 우회(w=4+4, 거리 0.5)가 빠름.
    solution = _build_solution(
        edges=[
            Edge(1, 2, 4, False),
            Edge(2, 3, 4, False),
            Edge(1, 3, 1, False),
            Edge(3, 1, 4, True),
        ],
        directed_edges={(1, 2), (2, 3), (1, 3)},
    )

    # 1→2:0.25, 1→3:0.5(우회), 2→3:0.25, 2→1:0.5, 3→1:0.25, 3→2:0.5 = 2.25
    assert ApspSumRanker(method="sum").run(solution) == pytest.approx(2.25)


def test_sum_keeps_fastest_parallel_edge_in_same_direction() -> None:
    solution = _build_solution(
        edges=[
            Edge(1, 2, 1, False),
            Edge(1, 2, 4, False),
            Edge(2, 1, 2, True),
        ],
        directed_edges={(1, 2)},
    )

    # 1→2 는 평행 간선 중 무거운 w=4(거리 0.25)만 유효, 2→1:0.5 = 0.75
    assert ApspSumRanker(method="sum").run(solution) == pytest.approx(0.75)


def test_sum_returns_inf_when_not_strongly_connected() -> None:
    solution = _build_solution(
        edges=[Edge(1, 2, 3, False), Edge(2, 3, 3, False)],
        directed_edges={(1, 2), (2, 3)},
    )

    assert ApspSumRanker(method="sum").run(solution) == float("inf")


def test_sum_raises_for_non_positive_weight() -> None:
    solution = _build_solution(
        edges=[Edge(1, 2, 0, False), Edge(2, 1, 1, True)],
        directed_edges={(1, 2)},
    )

    with pytest.raises(ValueError):
        ApspSumRanker(method="sum").run(solution)
