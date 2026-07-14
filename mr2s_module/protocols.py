from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from dimod import BinaryQuadraticModel
from dimod.higherorder.polynomial import BinaryPolynomial

if TYPE_CHECKING:
    from mr2s_module.domain.edge import Edge as EdgeModel
    from mr2s_module.domain.graph import Graph as GraphModel
    from mr2s_module.domain.graph_partition_result import (
        GraphPartitionResult as GraphPartitionResultModel,
    )
    from mr2s_module.domain.orientation_result import (
        OrientationResult as OrientationResultModel,
    )
    from mr2s_module.domain.embeddable_graph_partition import (
        EmbeddableGraphPartition as EmbeddableGraphPartitionModel,
    )
    from mr2s_module.domain.score import Score as ScoreModel
    from mr2s_module.domain.solution import Solution as SolutionModel
else:
    EdgeModel = Any
    GraphModel = Any
    GraphPartitionResultModel = Any
    OrientationResultModel = Any
    EmbeddableGraphPartitionModel = Any
    ScoreModel = Any
    SolutionModel = Any

Graph: TypeAlias = GraphModel
Edge: TypeAlias = EdgeModel
GraphPartitionResult: TypeAlias = GraphPartitionResultModel
OrientationResult: TypeAlias = OrientationResultModel
EmbeddableGraphPartition: TypeAlias = EmbeddableGraphPartitionModel
QuboMatrix: TypeAlias = BinaryQuadraticModel
Solution: TypeAlias = SolutionModel
Score: TypeAlias = ScoreModel


class FaceCycleProtocol(Protocol):
    def run(self, graph: Graph) -> GraphPartitionResult: ...


class EdgeOrientationProtocol(Protocol):
    """결과의 각 Edge 는 입력 graph 의 edge id 를 유지해야 한다 (Edge.oriented 참고)."""

    def run(self, graph: Graph) -> OrientationResult: ...


class DnCGraphPartitionStrategyProtocol(Protocol):
    def run(self, graph: Graph) -> EmbeddableGraphPartition: ...


class QuboSolverProtocol(Protocol):
    def run(self, qubo: QuboMatrix, graph: Graph) -> Solution: ...

    def run_with_embedding(
        self,
        qubo: QuboMatrix,
        graph: Graph,
        embedding: dict[object, list[object]],
    ) -> Solution: ...


class EvaluatorProtocol(Protocol):
    def run(self, solution: Solution) -> Score: ...


class Mr2sSolverProtocol(Protocol):
    # read-only property 로 선언해야 일반 속성·property 구현을 모두 허용한다.
    @property
    def evaluator(self) -> EvaluatorProtocol: ...
    def run(self, graph: Graph) -> Solution: ...


@runtime_checkable
class QuboBackedMr2sSolverProtocol(Mr2sSolverProtocol, Protocol):
    """QUBO 생성이 가능한 solver — 임베딩 기반 partition 전략이 요구하는 최소 표면.

    DnC 본체는 `run`/`evaluator` 만 쓰므로 이 프로토콜을 요구하지 않는다 (SA inner
    solver 도 허용). `build_bqm` 을 실제로 호출하는 전략에 넘기는 지점에서만 검사한다.
    """

    def build_bqm(self, graph: Graph) -> QuboMatrix: ...




class SolutionRankerProtocol(Protocol):
    def run(self, solution: Solution) -> float: ...


class PolyGeneratorProtocol(Protocol):
    def run(self, graph: Graph) -> BinaryPolynomial: ...
