from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from dimod import BinaryQuadraticModel, BinaryPolynomial

if TYPE_CHECKING:
    from mr2s_module.domain.edge import Edge as EdgeModel
    from mr2s_module.domain.graph import Graph as GraphModel
else:
    EdgeModel = Any
    GraphModel = Any

Graph: TypeAlias = GraphModel
Edge: TypeAlias = EdgeModel
QuboMatrix: TypeAlias = BinaryQuadraticModel
Solution: TypeAlias = set[tuple[int, int]]
Score: TypeAlias = float


class FaceCycleProtocol(Protocol):
    def run(self, graph: Graph) -> set[Edge]: ...


class QuboSolverProtocol(Protocol):
    def run(self, qubo: QuboMatrix, graph: Graph) -> Solution: ...


class EvaluatorProtocol(Protocol):
    def run(self, solution: Solution) -> Score: ...


class PolyGeneratorProtocol(Protocol):
    def run(self, graph: Graph) -> BinaryPolynomial: ...
