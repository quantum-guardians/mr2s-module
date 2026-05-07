import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.qubo import FlowPolyGenerator
from mr2s_module.util import estimate_required_qubits, map_binary_poly_to_bqm
from mr2s_module.util.embedding_util import EmbeddingEstimate


def _build_triangle_graph() -> Graph:
    return Graph(edges=[
        Edge(1, 2, 1, False),
        Edge(2, 3, 1, False),
        Edge(1, 3, 1, False),
    ])


def _build_5node_graph() -> Graph:
    """5노드 8간선 그래프."""
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


class TestMapBinaryPolyToBqm:
    def test_returns_binary_quadratic_model(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = map_binary_poly_to_bqm(polynomial)

        assert bqm is not None
        assert len(bqm.variables) > 0

    def test_variable_count_matches_undirected_edges(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = map_binary_poly_to_bqm(polynomial)

        assert len(bqm.variables) == 3

    def test_5node_graph_produces_expected_bqm_shape(self) -> None:
        graph = _build_5node_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = map_binary_poly_to_bqm(polynomial)

        assert len(bqm.variables) == 8
        assert len(bqm.quadratic) == 19


class TestEstimateRequiredQubits:
    @pytest.mark.slow
    def test_returns_embedding_estimate_for_triangle(self) -> None:
        graph = _build_triangle_graph()
        polynomial = FlowPolyGenerator().run(graph)
        bqm = map_binary_poly_to_bqm(polynomial)

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
        bqm = map_binary_poly_to_bqm(polynomial)

        result = estimate_required_qubits(bqm)

        assert isinstance(result, EmbeddingEstimate)
        assert result.num_logical_variables == len(bqm.variables)
        assert result.num_quadratic_couplings == len(bqm.quadratic)
        assert result.num_physical_qubits >= result.num_logical_variables
        assert result.max_chain_length >= 1
