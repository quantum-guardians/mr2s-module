from dimod import SimulatedAnnealingSampler

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol
from mr2s_module.qubo.solution_processing import select_best_sample


class SAQuboSolver:

  sampler = SimulatedAnnealingSampler()

  def __init__(self, ranker: SolutionRankerProtocol):
    self.ranker = ranker

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )
