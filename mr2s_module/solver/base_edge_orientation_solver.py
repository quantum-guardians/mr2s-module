from __future__ import annotations

from dimod import SampleSet

from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol, EdgeOrientationProtocol


class BaseEdgeOrientationSolver:
  """EdgeOrientationProtocol을 감싸 독립형 Mr2sSolver로 동작하게 하는 부모 클래스."""

  def __init__(
      self,
      edge_orienter: EdgeOrientationProtocol,
      evaluator: EvaluatorProtocol = Evaluator(),
  ) -> None:
    self.edge_orienter = edge_orienter
    self.evaluator = evaluator

  def run(self, graph: Graph) -> Solution:
    oriented_result = self.edge_orienter.run(graph)
    directed_edges = {edge.vertices for edge in oriented_result.get_edges()}

    # 검증: 프로토콜 결과 중 정렬 안 된(방향을 지정받지 못한) edge가 있는지 확인
    for edge in graph.edges.values():
      u, v = edge.endpoints()
      if (u, v) not in directed_edges and (v, u) not in directed_edges:
        raise ValueError(
          f"Edge {u}-{v} was not oriented by the algorithm. "
          f"This solver requires all edges to be oriented."
        )

    # sample_set 구축
    sample: dict[str, int] = {}
    for edge in graph.edges.values():
      u, v = edge.endpoints()
      if (u, v) in directed_edges:
        sample[edge.to_key()] = 0
      elif (v, u) in directed_edges:
        sample[edge.to_key()] = 1

    sample_set = SampleSet.from_samples(
      [sample],
      vartype="BINARY",
      energy=[0.0],
      num_occurrences=[1],
    )

    solution = Solution(
      edges=directed_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )
    solution.score = self.evaluator.run(solution)
    return solution
