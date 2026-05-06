from dimod import BinaryQuadraticModel

from mr2s_module.domain import Graph
from mr2s_module.qubo import FlowPolyGenerator
from mr2s_module.util import build_bqm, estimate_required_qubits
from mr2s_module.util.embedding_util import EmbeddingEstimate


class PolynomialOptimizationService:
    def __init__(self) -> None:
        self._flow_poly_generator = FlowPolyGenerator()

    def get_bqm(self, graph: Graph) -> BinaryQuadraticModel:
        """그래프로부터 다항식을 생성하고 BQM으로 변환하여 반환한다."""
        polynomial = self._flow_poly_generator.run(graph)
        return build_bqm(polynomial)

    def estimate(self, graph: Graph) -> EmbeddingEstimate:
        """그래프로부터 BQM을 생성하고 Pegasus P16 임베딩 기준 물리 큐빗 수를 추정한다."""
        bqm = self.get_bqm(graph)
        return estimate_required_qubits(bqm)
