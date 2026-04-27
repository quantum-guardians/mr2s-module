from mr2s_module.protocols import (
    EvaluatorProtocol,
    FaceCycleProtocol,
    Graph,
    QuboSolverProtocol,
    Score,
)


class MR2SSolver:
    def __init__(
        self,
        face_cycle: FaceCycleProtocol,
        qubo_solver: QuboSolverProtocol,
        evaluator: EvaluatorProtocol,
    ) -> None:
        self.face_cycle = face_cycle
        self.qubo_solver = qubo_solver
        self.evaluator = evaluator

    def run(self, graph: Graph) -> Score:
        raise NotImplementedError
