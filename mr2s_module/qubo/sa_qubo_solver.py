from dimod import SimulatedAnnealingSampler, SampleSet

from mr2s_module.domain import Graph
from mr2s_module.protocols import QuboMatrix, Solution, EvaluatorProtocol, Edge
from mr2s_module.qubo import QuboSolver


class SAQuboSolver(QuboSolver):

  sampler = SimulatedAnnealingSampler()

  def __init__(self, evaluator: EvaluatorProtocol):
    self.evaluator = evaluator

  @staticmethod
  def _process_solution(
      best_sample: dict[str, int], canonical_edges: list[Edge]
  ) -> set[tuple[int, int]]:
    """
    Processes the best sample from the solver into a list of directed edge tuples.

    Returns:
        list[tuple[int, int]]: Directed edges represented as (u, v) integer node ID pairs.
    """
    final_edges = set()
    for edge in canonical_edges:

      # handle predefined edges
      if edge.directed:
        final_edges.add(edge.vertices)
        continue

      # handle optimized edges
      var_name = edge.to_key()

      if best_sample.get(var_name, 0) == 1:
        final_edges.add((edge.vertices[1], edge.vertices[0]))
      else:
        final_edges.add((edge.vertices[0], edge.vertices[1]))

    return final_edges

  def _select_best_sample(
      self,
      sample_set: SampleSet,
      canonical_edges: list[Edge],
  ) -> set[tuple[int, int]]:

    def get_effective_score(tuples: set[tuple[int, int]]):
      score = self.evaluator.run(set(tuples))
      return float('inf') if score == -1 else score

    return min(
      map(lambda sample: self._process_solution(sample, canonical_edges), sample_set.samples()),
      key=get_effective_score
    )

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo)
    return self._select_best_sample(sample_set, graph.edges)
