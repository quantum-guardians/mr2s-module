"""QUBO 솔버 공통 해(sample) 후처리 유틸리티.

어닐러(`QuboSolver`)가 반환한 `SampleSet`을 실제 간선
방향(directed edge) 집합으로 변환하고, 여러 sample 중 ranker 기준 최적해를
고르는 로직을 모아둔다. 솔버 간 코드 중복을 피하기 위해 모듈 레벨 free
function 으로 제공한다.
"""

from dimod import SampleSet

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import Edge, SolutionRankerProtocol


def safe_lookup(sample, var_name: str) -> int:
  # dimod SampleView.get() raises ValueError on unknown vars (Mapping.get
  # only catches KeyError), so 다항식 합성 중 계수가 0 으로 사라져 BQM 에
  # 등록되지 않은 변수는 default 0 으로 처리한다.
  try:
    return int(sample[var_name])
  except (KeyError, ValueError):
    return 0


def process_solution(
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
    bit = safe_lookup(best_sample, var_name)

    if bit == 1:
      final_edges.add((edge.vertices[1], edge.vertices[0]))
    else:
      final_edges.add((edge.vertices[0], edge.vertices[1]))

  return final_edges


def select_best_sample(
    sample_set: SampleSet,
    canonical_edges: list[Edge],
    ranker: SolutionRankerProtocol,
) -> set[tuple[int, int]]:
  if len(sample_set) == 0:
    return process_solution({}, canonical_edges)

  def get_effective_score(tuples: set[tuple[int, int]]):
    solution = Solution(
      edges=tuples,
      graph=Graph(edges=canonical_edges),
      sample_set=sample_set,
      score=None,
    )

    return ranker.run(solution)

  return min(
    map(lambda sample: process_solution(sample, canonical_edges), sample_set.samples()),
    key=get_effective_score
  )
