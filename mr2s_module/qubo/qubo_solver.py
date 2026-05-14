from mr2s_module.protocols import QuboMatrix, Solution, Graph, QuboSolverProtocol


class QuboSolver(QuboSolverProtocol):
    def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
        raise NotImplementedError
