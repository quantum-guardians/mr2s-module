import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.qubo import FlowPolyGenerator
from mr2s_module.service import PolynomialOptimizationService
from mr2s_module.util import build_bqm, estimate_required_qubits
from mr2s_module.util.embedding_util import EmbeddingEstimate


def _build_triangle_graph() -> Graph:
    return Graph(edges=[
        Edge(1, 2, 1, False),
        Edge(2, 3, 1, False),
        Edge(1, 3, 1, False),
    ])


def _build_5node_graph() -> Graph:
    """5노드 8간선 그래프 (이슈 테스트 케이스와 동일)."""
    return Graph(edges=[
        Edge(1, 2, 1, False),
        Edge(1, 3, 1, False),
        Edge(1, 4, 1, False),
        Edge(2, 3, 1, False),
        Edge(2, 4, 1, False),
        Edge(3, 4, 1, False),
        Edge(3, 5, 1, False),
        Edge(4, 5, 1, False),
    ])


class TestBuildBqm:
    def test_build_bqm_returns_binary_quadratic_model(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = build_bqm(polynomial)

        assert bqm is not None
        assert len(bqm.variables) > 0

    def test_build_bqm_variable_count_matches_undirected_edges(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = build_bqm(polynomial)

        assert len(bqm.variables) == 3


class TestEstimateRequiredQubits:
    @pytest.mark.slow
    def test_returns_embedding_estimate_for_triangle(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = build_bqm(polynomial)

        result = estimate_required_qubits(bqm)

        assert isinstance(result, EmbeddingEstimate)
        assert result.num_logical_variables == len(bqm.variables)
        assert result.num_quadratic_couplings == len(bqm.quadratic)
        assert result.num_physical_qubits >= result.num_logical_variables
        assert result.max_chain_length >= 1

    @pytest.mark.slow
    def test_returns_embedding_estimate_for_5node_graph(self) -> None:
        graph = _build_5node_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = build_bqm(polynomial)

        result = estimate_required_qubits(bqm)

        assert isinstance(result, EmbeddingEstimate)
        assert result.num_logical_variables == len(bqm.variables)
        assert result.num_quadratic_couplings == len(bqm.quadratic)
        assert result.num_physical_qubits >= result.num_logical_variables
        assert result.max_chain_length >= 1


class TestPolynomialOptimizationService:
    def test_get_bqm_returns_valid_bqm(self) -> None:
        service = PolynomialOptimizationService()
        graph = _build_triangle_graph()

        bqm = service.get_bqm(graph)

        assert bqm is not None
        assert len(bqm.variables) == 3
        assert len(bqm.quadratic) > 0

    def test_get_bqm_5node_graph(self) -> None:
        service = PolynomialOptimizationService()
        graph = _build_5node_graph()

        bqm = service.get_bqm(graph)

        assert bqm is not None
        assert len(bqm.variables) == 8
        assert len(bqm.quadratic) == 19

    @pytest.mark.slow
    def test_estimate_returns_embedding_estimate(self) -> None:
        service = PolynomialOptimizationService()
        graph = _build_5node_graph()

        result = service.estimate(graph)

        assert isinstance(result, EmbeddingEstimate)
        assert result.num_logical_variables == 8
        assert result.num_quadratic_couplings == 19
        assert result.num_physical_qubits >= 8
        assert result.max_chain_length >= 1
