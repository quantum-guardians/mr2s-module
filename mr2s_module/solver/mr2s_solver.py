from mr2s_module.protocols import (
    EdgeOrientationProtocol,
    EvaluatorProtocol,
    Graph,
    QuboSolverProtocol,
    Score,
)


class MR2SSolver:
    def __init__(
        self,
        edge_orienter: EdgeOrientationProtocol,
        qubo_solver: QuboSolverProtocol,
        evaluator: EvaluatorProtocol,
    ) -> None:
        self.edge_orienter = edge_orienter
        self.qubo_solver = qubo_solver
        self.evaluator = evaluator

    def run(self, graph: Graph) -> Score:
        raise NotImplementedError
