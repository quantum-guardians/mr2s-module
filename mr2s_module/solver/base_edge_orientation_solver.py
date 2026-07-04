from __future__ import annotations

from dimod import SampleSet

from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol, EdgeOrientationProtocol


class BaseEdgeOrientationSolver:
  """EdgeOrientationProtocol을 감싸 독립형 Mr2sSolver로 동작하게 하는 부모 클래스.

  orienter 결과 Edge 는 도메인 edge id 를 유지해야 한다(`Edge.oriented()` 참고).
  id 로 방향을 복원하므로 평행 간선도 copy 별 독립 방향이 보존된다.
  """

  def __init__(
      self,
      edge_orienter: EdgeOrientationProtocol,
      evaluator: EvaluatorProtocol = Evaluator(),
  ) -> None:
    self.edge_orienter = edge_orienter
    self.evaluator = evaluator

  def run(self, graph: Graph) -> Solution:
    oriented_result = self.edge_orienter.run(graph)
    directed_by_id = {edge.id: edge.vertices for edge in oriented_result.get_edges()}

    # 검증: 프로토콜 결과 중 정렬 안 된(방향을 지정받지 못한) edge가 있는지 확인
    for edge in graph.edges.values():
      if edge.id not in directed_by_id:
        u, v = edge.endpoints()
        raise ValueError(
          f"Edge {u}-{v} (id={edge.id}) was not oriented by the algorithm. "
          f"This solver requires all edges to be oriented."
        )

    # sample_set + edge id → 방향 매핑 구축
    sample: dict[str, int] = {}
    solution_edges: dict[int, tuple[int, int]] = {}
    for edge in graph.edges.values():
      direction = directed_by_id[edge.id]
      sample[edge.to_key()] = 0 if direction == edge.endpoints() else 1
      solution_edges[edge.id] = direction

    sample_set = SampleSet.from_samples(
      [sample],
      vartype="BINARY",
      energy=[0.0],
      num_occurrences=[1],
    )

    solution = Solution(
      edges=solution_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )
    solution.score = self.evaluator.run(solution)
    return solution
