import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.solver.base_edge_orientation_solver import BaseEdgeOrientationSolver


class StubPartialEdgeOrienter:
    """일부 또는 모든 간선 방향을 결정하는 스텁 orienter."""

    def __init__(self, oriented_edges: list[Edge]) -> None:
        self.oriented_edges = oriented_edges

    def run(self, graph: Graph) -> OrientedEdges:
        return OrientedEdges(edges=self.oriented_edges)


def test_base_solver_raises_error_when_edge_not_oriented() -> None:
    # 1-2, 2-3 간선이 있지만, 2-3 간선만 방향이 정해지고 1-2 간선은 정해지지 않은 케이스
    edge_ab = Edge(1, 2, 1, False)
    edge_bc = Edge(2, 3, 1, False)
    graph = Graph(edges=[edge_ab, edge_bc])

    oriented_edges = [
        edge_bc.oriented(2, 3),  # 2->3만 방향 결정됨
    ]

    solver = BaseEdgeOrientationSolver(
        edge_orienter=StubPartialEdgeOrienter(oriented_edges)
    )

    # 1-2 간선이 누락되었으므로 ValueError가 발생해야 함
    with pytest.raises(ValueError) as excinfo:
        solver.run(graph)

    assert "was not oriented by the algorithm" in str(excinfo.value)
    assert "This solver requires all edges to be oriented." in str(excinfo.value)


def test_base_solver_succeeds_when_all_edges_oriented() -> None:
    # 모든 간선이 정해진 케이스
    edge_ab = Edge(1, 2, 1, False)
    edge_bc = Edge(2, 3, 1, False)
    graph = Graph(edges=[edge_ab, edge_bc])

    oriented_edges = [
        edge_ab.oriented(1, 2),
        edge_bc.oriented(2, 3),
    ]

    solver = BaseEdgeOrientationSolver(
        edge_orienter=StubPartialEdgeOrienter(oriented_edges)
    )

    solution = solver.run(graph)
    assert len(solution.edges) == 2
    assert solution.score is not None


def test_base_solver_preserves_anti_parallel_pair_per_edge_id() -> None:
    # 평행쌍이 반대 방향으로 결정된 경우 id 별 방향이 붕괴 없이 보존돼야 함
    edge_a = Edge(0, 1, 3, False)
    edge_b = Edge(0, 1, 3, False)
    graph = Graph(edges=[edge_a, edge_b])

    oriented_edges = [
        edge_a.oriented(0, 1),
        edge_b.oriented(1, 0),
    ]

    solver = BaseEdgeOrientationSolver(
        edge_orienter=StubPartialEdgeOrienter(oriented_edges)
    )

    solution = solver.run(graph)
    assert solution.edges[edge_a.id] == (0, 1)
    assert solution.edges[edge_b.id] == (1, 0)
    assert solution.score is not None
    assert solution.score.flow_score == 0.0
    assert solution.score.strong_connect_rate == 1.0
