from dimod import SimulatedAnnealingSampler, SampleSet

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol, Edge
from mr2s_module.qubo import QuboSolver


class SAQuboSolver(QuboSolver):

  sampler = SimulatedAnnealingSampler()

  def __init__(self, ranker: SolutionRankerProtocol):
    self.ranker = ranker

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
      bit = SAQuboSolver._safe_lookup(best_sample, var_name)

      if bit == 1:
        final_edges.add((edge.vertices[1], edge.vertices[0]))
      else:
        final_edges.add((edge.vertices[0], edge.vertices[1]))

    return final_edges

  @staticmethod
  def _safe_lookup(sample, var_name: str) -> int:
    # dimod SampleView.get() raises ValueError on unknown vars (Mapping.get
    # only catches KeyError), so 다항식 합성 중 계수가 0 으로 사라져 BQM 에
    # 등록되지 않은 변수는 default 0 으로 처리한다.
    try:
      return int(sample[var_name])
    except (KeyError, ValueError):
      return 0

  def _select_best_sample(
      self,
      sample_set: SampleSet,
      canonical_edges: list[Edge],
  ) -> set[tuple[int, int]]:

    def get_effective_score(tuples: set[tuple[int, int]]):
      solution = Solution(
        edges=tuples,
        graph=Graph(edges=canonical_edges),
        sample_set=sample_set,
        score=None,
      )

      try:
        return self.ranker.run(solution)
      except AssertionError:
        return float("inf")

    return min(
      map(lambda sample: self._process_solution(sample, canonical_edges), sample_set.samples()),
      key=get_effective_score
    )

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo)
    return Solution(
      edges=self._select_best_sample(sample_set, graph.edges),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )
