from mr2s_module.protocols import QuboMatrix, Solution


class QuboSolver:
    def run(self, qubo: QuboMatrix) -> Solution:
        raise NotImplementedError
