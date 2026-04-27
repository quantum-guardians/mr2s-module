from mr2s_module.protocols import Score, Solution


class Evaluator:
    def run(self, solution: Solution) -> Score:
        raise NotImplementedError
