import time

import networkx as nx
import pytest

from mr2s_module.edge_orient.robbin import Robbin
from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import empty_binary_sample_set
from tests.cycle.conftest import remove_edges_by_percent
from tests.util.graph_fixtures import delaunay_graph, graph_from_pairs


def _apply_cycle_directions(graph: Graph, cycle: Robbin) -> None:
    """Cycle 이 결정한 방향을 graph 에 박아 solver 가 보존하도록 강제한다."""
    graph.define_edge_direction(set(cycle.run(graph).get_edges()))


def test_robbin_basic_cases():
    graph = graph_from_pairs([(0, 1), (1, 2), (0, 2)])
    directed_edges = Robbin().run(graph).get_edges()
    assert len(directed_edges) == 3
    assert all(e.directed for e in directed_edges)
    dg = nx.DiGraph()
    for e in directed_edges:
        dg.add_edge(*e.vertices)
    assert nx.is_strongly_connected(dg)


def test_robbin_bridge_falls_back_to_no_orientation():
    graph = graph_from_pairs([(0, 1), (1, 2)])
    directed_edges = Robbin().run(graph).get_edges()
    assert directed_edges == []


def test_robbin_integration_with_real_solver():
    # 삼각형: cycle 이 모든 간선을 방향 결정 → SA 는 이를 보존해야 함.
    graph = graph_from_pairs([(0, 1), (1, 2), (0, 2)])
    cycle = Robbin()
    _apply_cycle_directions(graph, cycle)

    expected_directions = {e.vertices for e in graph.edges.values() if e.directed}
    assert len(expected_directions) == 3

    solver = DnCMr2sSolver(mr2s_solver=QuboMR2SSolver())
    solution = solver.run(graph)

    assert isinstance(solution, DnCSolution)
    assert len(solution.edges) == 3
    assert solution.edges == expected_directions


@pytest.mark.slow
@pytest.mark.parametrize("num_points", [100, 500])
@pytest.mark.parametrize("remove_percent", [0, 20, 50])
def test_robbin_performance_and_apsp(num_points, remove_percent):
    base_graph = delaunay_graph(n=num_points, seed=42)
    graph, removed_count = remove_edges_by_percent(base_graph, remove_percent)
    if graph.is_empty():
        return

    cycle = Robbin()

    start_time = time.perf_counter()
    directed_edges = cycle.run(graph).get_edges()
    elapsed = time.perf_counter() - start_time

    oriented_edges = [e for e in directed_edges if e.directed]
    oriented_ids = {e.id for e in oriented_edges}
    undirected_edges = [e for e in graph.edges.values() if e.id not in oriented_ids]

    # Cycle 방향을 graph 에 박은 뒤 실제 솔버 통과.
    graph.define_edge_direction(set(directed_edges))
    solver = DnCMr2sSolver(mr2s_solver=QuboMR2SSolver())
    solution = solver.run(graph)

    evaluator = Evaluator()
    partial_sol = Solution(
        edges={(e.vertices[0], e.vertices[1]) for e in oriented_edges},
        graph=Graph(edges=oriented_edges + undirected_edges),
        sample_set=empty_binary_sample_set(),
    )
    partial_apsp = evaluator.eval_apsp_sum(partial_sol)
    final_apsp = evaluator.eval_apsp_sum(solution)

    print(f"\nRobbin Performance (n={num_points}, remove={remove_percent}%)")
    print(f"  elapsed_seconds: {elapsed:.4f}")
    print(f"  original_edges: {len(base_graph.edges)}")
    print(f"  removed_edges: {removed_count}")
    print(f"  final_edges: {len(graph.edges)}")
    print(f"  directed_by_algo: {len(oriented_edges)}")
    print(f"  undirected_by_algo: {len(undirected_edges)}")
    print(f"  partial_apsp: {partial_apsp}")
    print(f"  final_apsp (after solver): {final_apsp}")

    assert final_apsp < float("inf")
    # Cycle 이 결정한 방향은 solver 통과 후에도 그대로 살아있어야 함.
    expected_directions = {e.vertices for e in graph.edges.values() if e.directed}
    assert expected_directions.issubset(solution.edges)
