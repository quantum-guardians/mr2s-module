from typing import Any, Protocol, TypeAlias

Graph: TypeAlias = Any
Edge: TypeAlias = Any
QuboMatrix: TypeAlias = Any
Solution: TypeAlias = Any
Score: TypeAlias = float


class FaceCycleProtocol(Protocol):
    def run(self, graph: Graph) -> set[Edge]: ...


class QuboSolverProtocol(Protocol):
    def run(self, qubo: QuboMatrix) -> Solution: ...


class EvaluatorProtocol(Protocol):
    def run(self, solution: Solution) -> Score: ...
