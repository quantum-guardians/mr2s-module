from mr2s_module.protocols import QuboMatrix, Solution, Graph


class QuboSolver:
    def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
        raise NotImplementedError
