from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from dimod import BinaryQuadraticModel, BinaryPolynomial

if TYPE_CHECKING:
    from mr2s_module.domain.edge import Edge as EdgeModel
    from mr2s_module.domain.graph import Graph as GraphModel
    from mr2s_module.domain.score import Score as ScoreModel
    from mr2s_module.domain.solution import Solution as SolutionModel
else:
    EdgeModel = Any
    GraphModel = Any
    ScoreModel = Any
    SolutionModel = Any

Graph: TypeAlias = GraphModel
Edge: TypeAlias = EdgeModel
QuboMatrix: TypeAlias = BinaryQuadraticModel
Solution: TypeAlias = SolutionModel
Score: TypeAlias = ScoreModel


class FaceCycleProtocol(Protocol):
    def run(self, graph: Graph) -> set[Edge]: ...


class QuboSolverProtocol(Protocol):
    def run(self, qubo: QuboMatrix, graph: Graph) -> Solution: ...


class EvaluatorProtocol(Protocol):
    def run(self, solution: Solution) -> Score: ...


class SolutionRankerProtocol(Protocol):
    def run(self, solution: Solution) -> float: ...


class PolyGeneratorProtocol(Protocol):
    def run(self, graph: Graph) -> BinaryPolynomial: ...
