from typing import Protocol, TypeAlias

from dimod import BinaryQuadraticModel, BinaryPolynomial

from mr2s_module.domain import Edge, Graph

Graph: TypeAlias = Graph
Edge: TypeAlias = Edge
QuboMatrix: TypeAlias = BinaryQuadraticModel
Solution: TypeAlias = set[tuple[int, int]]
Score: TypeAlias = float


class FaceCycleProtocol(Protocol):
    def run(self, graph: Graph) -> set[Edge]: ...


class QuboSolverProtocol(Protocol):
    def run(self, qubo: QuboMatrix) -> Solution: ...


class EvaluatorProtocol(Protocol):
    def run(self, solution: Solution) -> Score: ...
