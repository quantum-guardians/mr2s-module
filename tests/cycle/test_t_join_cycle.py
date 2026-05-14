import time

import pytest

from mr2s_module.cycle.t_join_cycle import TjoinCycle
from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import empty_binary_sample_set
from tests.cycle.conftest import remove_edges_by_percent
from tests.util.graph_fixtures import delaunay_graph, graph_from_pairs


def _apply_cycle_directions(graph: Graph, cycle: TjoinCycle) -> None:
    """Cycle 이 결정한 방향을 graph 에 박아 solver 가 보존하도록 강제한다."""
    result = cycle.run(graph)
    graph.define_edge_direction({e for e in result.directed_edges()})


def test_t_join_cycle_basic_cases():
    # Triangle: 모든 차수 짝수 → 전부 oriented.
    graph = graph_from_pairs([(0, 1), (1, 2), (0, 2)])
    result = TjoinCycle().run(graph)
    assert len(result.sub_graphs) == 1
    assert len(result.sub_graphs[0].edges) == 3
    assert len(result.remaining_edges) == 0

    # Path: 모든 간선이 T-join 으로 빠져 oriented 없음.
    graph = graph_from_pairs([(0, 1), (1, 2)])
    result = TjoinCycle().run(graph)
    assert len(result.sub_graphs[0].edges) == 0
    assert len(result.remaining_edges) == 2


def test_t_join_cycle_integration_with_real_solver():
    # 사각형 + 꼬리: 사각형 4개는 oriented, 꼬리 1개는 T-join 으로 빠짐.
    graph = graph_from_pairs([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)])
    cycle = TjoinCycle()
    _apply_cycle_directions(graph, cycle)

    expected_directions = {e.vertices for e in graph.edges if e.directed}
    assert len(expected_directions) == 4

    solver = DnCMr2sSolver(mr2s_solver=QuboMR2SSolver(), face_cycle=cycle)
    solution = solver.run(graph)

    assert isinstance(solution, DnCSolution)
    assert len(solution.edges) == 5
    # Cycle 이 결정한 방향은 solver 가 뒤집지 않음.
    assert expected_directions.issubset(solution.edges)


@pytest.mark.slow
@pytest.mark.parametrize("num_points", [100, 500])
@pytest.mark.parametrize("remove_percent", [0, 20, 50])
def test_t_join_cycle_performance_and_apsp(num_points, remove_percent):
    base_graph = delaunay_graph(n=num_points, seed=42)
    graph, removed_count = remove_edges_by_percent(base_graph, remove_percent)
    if graph.is_empty():
        return

    cycle = TjoinCycle()

    start_time = time.perf_counter()
    result = cycle.run(graph)
    elapsed = time.perf_counter() - start_time

    directed_edges = [e for sg in result.sub_graphs for e in sg.edges]

    graph.define_edge_direction({e for e in result.directed_edges()})
    solver = DnCMr2sSolver(mr2s_solver=QuboMR2SSolver(), face_cycle=cycle)
    solution = solver.run(graph)

    evaluator = Evaluator()
    partial_sol = Solution(
        edges={(e.vertices[0], e.vertices[1]) for e in directed_edges},
        graph=Graph(edges=directed_edges + result.remaining_edges),
        sample_set=empty_binary_sample_set(),
    )
    partial_apsp = evaluator.eval_apsp_sum(partial_sol)
    final_apsp = evaluator.eval_apsp_sum(solution)

    print(f"\nTjoinCycle Performance (n={num_points}, remove={remove_percent}%)")
    print(f"  elapsed_seconds: {elapsed:.4f}")
    print(f"  original_edges: {len(base_graph.edges)}")
    print(f"  removed_edges: {removed_count}")
    print(f"  final_edges: {len(graph.edges)}")
    print(f"  directed_by_algo: {len(directed_edges)}")
    print(f"  remaining_undirected: {len(result.remaining_edges)}")
    print(f"  partial_apsp: {partial_apsp}")
    print(f"  final_apsp (after solver): {final_apsp}")

    assert final_apsp < float("inf")
    expected_directions = {e.vertices for e in graph.edges if e.directed}
    assert expected_directions.issubset(solution.edges)
