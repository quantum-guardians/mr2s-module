from dwave.system import DWaveSampler, EmbeddingComposite

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol
from mr2s_module.qubo.solution_processing import select_best_sample


class QAQuboSolver:

  def __init__(self, ranker: SolutionRankerProtocol, num_reads: int = 100):
    # num_reads: 어닐러를 몇 번 반복 샘플링할지. 기본 100
    self.ranker = ranker
    self.num_reads = num_reads
    try:
      self.sampler = EmbeddingComposite(DWaveSampler())
    except Exception as e:
      raise RuntimeError(
        "D-Wave API credentials not found. "
        "Set DWAVE_API_TOKEN environment variable or configure "
        "~/.config/dwave/dwave.conf"
      ) from e

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo, num_reads=self.num_reads)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )
