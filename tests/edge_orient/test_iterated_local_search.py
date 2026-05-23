import time

import networkx as nx
import numpy as np
import pytest

from mr2s_module.domain import Edge, Graph, OrientedEdges
from mr2s_module.edge_orient.iterated_local_search import (
    IteratedLocalSearch,
    evaluate_score,
    n1_search,
    n2_search,
    n3_search,
)
from mr2s_module.domain.orientation import Orientation
from mr2s_module.evaluator import Evaluator
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from tests.cycle.conftest import remove_edges_by_percent
from tests.util.graph_fixtures import delaunay_graph, graph_from_pairs


class TestEvaluateScore:
    def test_empty_graph_returns_inf(self):
        assert evaluate_score(Orientation(), []) == float('inf')

    def test_disconnected_returns_inf(self):
        o = Orientation({frozenset({0, 1}): (0, 1)})
        assert evaluate_score(o, [0, 1, 2]) == float('inf')

    def test_not_strongly_connected_returns_inf(self):
        o = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (0, 2),
        })
        assert evaluate_score(o, [0, 1, 2]) == float('inf')

    def test_directed_cycle_score(self):
        o = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        assert evaluate_score(o, [0, 1, 2]) == 9.0

    def test_weighted_cycle_score_with_unit_weights(self):
        o = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        weight_map = {
            frozenset({0, 1}): 1,
            frozenset({1, 2}): 1,
            frozenset({0, 2}): 1,
        }
        assert evaluate_score(o, [0, 1, 2], weight_map) == 9.0

    def test_weighted_score_differs_from_unweighted(self):
        o = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        weight_map = {
            frozenset({0, 1}): 1,
            frozenset({1, 2}): 1,
            frozenset({0, 2}): 100,
        }
        # 0->1: 1, 0->2: 1+1=2, 1->2: 1, 1->0: 1+100=101, 2->0: 100, 2->1: 100+1=101
        # total = 1+2+1+101+100+101 = 306
        assert evaluate_score(o, [0, 1, 2], weight_map) == 306.0

    def test_weighted_disconnected_returns_inf(self):
        o = Orientation({frozenset({0, 1}): (0, 1), frozenset({1, 2}): (1, 2)})
        weight_map = {frozenset({0, 1}): 10, frozenset({1, 2}): 10}
        assert evaluate_score(o, [0, 1, 2], weight_map) == float('inf')


class TestN1Search:
    def test_improves_bad_orientation(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        bad = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (0, 2),
        })
        snapshot = bad.copy()
        new_orient, new_score = n1_search(bad, float('inf'), base, [0, 1, 2], None, rng)
        assert new_score < float('inf')
        assert new_orient != snapshot
        assert bad == snapshot  # 입력은 변형되지 않아야 함

    def test_no_improvement_returns_original(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        optimal = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        score = 9.0
        new_orient, new_score = n1_search(optimal, score, base, [0, 1, 2], None, rng)
        assert new_score == score
        assert new_orient == optimal


class TestN2Search:
    def test_improves_bad_orientation(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        bad = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (0, 2),
        })
        snapshot = bad.copy()
        new_orient, new_score = n2_search(bad, float('inf'), base, [0, 1, 2], None, rng)
        assert new_score < float('inf')
        assert new_orient != snapshot
        assert bad == snapshot

    def test_no_improvement_returns_original(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        optimal = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        score = 9.0
        new_orient, new_score = n2_search(optimal, score, base, [0, 1, 2], None, rng)
        assert new_score == score
        assert new_orient == optimal


class TestN3Search:
    def test_with_cyclic_graph_returns_sc_orientation(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
        orient = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({2, 3}): (2, 3),
            frozenset({3, 0}): (3, 0),
        })
        score = evaluate_score(orient, [0, 1, 2, 3])
        assert score < float('inf')
        new_orient, new_score = n3_search(orient, score, base, [0, 1, 2, 3], None, rng)
        assert new_score < float('inf')

    def test_with_dag_orientation_returns_original(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        dag = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (0, 2),
        })
        new_orient, new_score = n3_search(dag, float('inf'), base, [0, 1, 2], None, rng)
        assert new_score == float('inf')
        assert new_orient == dag

    def test_acyclic_graph_returns_original(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2)])
        tree_orient = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
        })
        score = float('inf')
        new_orient, new_score = n3_search(tree_orient, score, base, [0, 1, 2], None, rng)
        assert new_score == score
        assert new_orient == tree_orient

    def test_no_improvement_returns_original(self):
        rng = np.random.default_rng(0)
        base = nx.Graph([(0, 1), (1, 2), (0, 2)])
        optimal = Orientation({
            frozenset({0, 1}): (0, 1),
            frozenset({1, 2}): (1, 2),
            frozenset({0, 2}): (2, 0),
        })
        score = 9.0
        new_orient, new_score = n3_search(optimal, score, base, [0, 1, 2], None, rng)
        assert new_score == score
        assert new_orient == optimal


class TestIteratedLocalSearch:
    def test_empty_graph_returns_empty_edges(self):
        result = IteratedLocalSearch().run(graph_from_pairs([]))
        assert isinstance(result, OrientedEdges)
        assert result.get_edges() == []

    def test_triangle_returns_strongly_connected(self):
        result = IteratedLocalSearch().run(graph_from_pairs([(0, 1), (1, 2), (0, 2)]))
        edges = result.get_edges()
        assert len(edges) == 3
        assert all(e.directed for e in edges)

        dg = nx.DiGraph()
        for e in edges:
            dg.add_edge(*e.vertices)
        assert nx.is_strongly_connected(dg)

    def test_square_returns_strongly_connected(self):
        result = IteratedLocalSearch().run(
            graph_from_pairs([(0, 1), (1, 2), (2, 3), (0, 3)])
        )
        edges = result.get_edges()
        assert len(edges) == 4
        assert all(e.directed for e in edges)

        dg = nx.DiGraph()
        for e in edges:
            dg.add_edge(*e.vertices)
        assert nx.is_strongly_connected(dg)

    def test_k4_returns_strongly_connected(self):
        pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        ]
        result = IteratedLocalSearch().run(graph_from_pairs(pairs))
        edges = result.get_edges()
        assert len(edges) == 6
        assert all(e.directed for e in edges)

        dg = nx.DiGraph()
        for e in edges:
            dg.add_edge(*e.vertices)
        assert nx.is_strongly_connected(dg)

    def test_k5_returns_strongly_connected(self):
        pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (3, 4),
        ]
        result = IteratedLocalSearch().run(graph_from_pairs(pairs))
        edges = result.get_edges()
        assert len(edges) == 10
        assert all(e.directed for e in edges)

        dg = nx.DiGraph()
        for e in edges:
            dg.add_edge(*e.vertices)
        assert nx.is_strongly_connected(dg)

    def test_bridge_returns_orientation_with_all_edges(self):
        result = IteratedLocalSearch().run(graph_from_pairs([(0, 1)]))
        edges = result.get_edges()
        assert len(edges) == 1
        assert edges[0].directed

    def test_preserves_edge_weights(self):
        graph = Graph(edges=[
            Edge(0, 1, weight=5, directed=False),
            Edge(1, 2, weight=3, directed=False),
            Edge(0, 2, weight=2, directed=False),
        ])
        result = IteratedLocalSearch().run(graph)
        edges = result.get_edges()
        weight_map = {frozenset(e.id): e.weight for e in edges}
        assert weight_map[frozenset({0, 1})] == 5
        assert weight_map[frozenset({1, 2})] == 3
        assert weight_map[frozenset({0, 2})] == 2

    def test_early_stopping_with_patience_1(self):
        ils = IteratedLocalSearch(max_iter=100, patience=1)
        result = ils.run(graph_from_pairs([(0, 1), (1, 2), (0, 2)]))
        assert isinstance(result, OrientedEdges)
        assert len(result.get_edges()) == 3

    def test_max_iter_1(self):
        ils = IteratedLocalSearch(max_iter=1, patience=100)
        result = ils.run(graph_from_pairs([(0, 1), (1, 2), (0, 2)]))
        assert isinstance(result, OrientedEdges)
        assert len(result.get_edges()) == 3

    @pytest.mark.parametrize("n_verts", [4, 6, 8])
    def test_cycle_graphs(self, n_verts):
        pairs = [(i, (i + 1) % n_verts) for i in range(n_verts)]
        result = IteratedLocalSearch().run(graph_from_pairs(pairs))
        edges = result.get_edges()
        assert len(edges) == n_verts
        dg = nx.DiGraph()
        for e in edges:
            dg.add_edge(*e.vertices)
        assert nx.is_strongly_connected(dg)


@pytest.mark.slow
def test_ils_integration_with_sa_solver():
    """약 300개 정점 Delaunay 그래프에 ILS 를 돌리고 SA 솔버까지 통과시켜 동작 확인.

    ILS 가 모든 간선을 방향 결정해 두므로 SA 의 anneal 루프는 빈 variable_edges 로
    즉시 빠진다. 본 테스트는 통합 경로(Solution 생성, Evaluator 평가)를 검증한다.
    """
    base_graph = delaunay_graph(n=300, seed=42)
    graph, removed_count = remove_edges_by_percent(base_graph, 0)
    if graph.is_empty():
        return

    ils = IteratedLocalSearch(max_iter=30, patience=5)

    start_time = time.perf_counter()
    directed_edges = ils.run(graph).get_edges()
    elapsed = time.perf_counter() - start_time

    assert len(directed_edges) == len(graph.edges)
    assert all(e.directed for e in directed_edges)

    dg = nx.DiGraph()
    for e in directed_edges:
        dg.add_edge(*e.vertices)
    assert nx.is_strongly_connected(dg), "ILS 결과는 강연결이어야 한다"

    graph.define_edge_direction(set(directed_edges))
    solver = SAMR2SSolver()
    solution = solver.run(graph)

    final_apsp = Evaluator().eval_apsp_sum(solution)

    print(f"\nILS Integration with SA (n=300, remove=0%)")
    print(f"  ils_elapsed_seconds: {elapsed:.4f}")
    print(f"  original_edges: {len(base_graph.edges)}")
    print(f"  removed_edges: {removed_count}")
    print(f"  final_edges: {len(graph.edges)}")
    print(f"  directed_by_ils: {len(directed_edges)}")
    print(f"  final_apsp (after SA solver): {final_apsp}")

    assert final_apsp < float("inf")
    # ILS 가 결정한 방향은 SA 통과 후에도 그대로 살아있어야 함.
    expected_directions = {e.vertices for e in graph.edges if e.directed}
    assert expected_directions.issubset(solution.edges)
