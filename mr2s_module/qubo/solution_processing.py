"""QUBO 솔버 공통 해(sample) 후처리 유틸리티.

어닐러(`QuboSolver`)가 반환한 `SampleSet`을 실제 간선
방향(directed edge) 집합으로 변환하고, 여러 sample 중 ranker 기준 최적해를
고르는 로직을 모아둔다. 솔버 간 코드 중복을 피하기 위해 모듈 레벨 free
function 으로 제공한다.
"""

from dimod import SampleSet

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.protocols import SolutionRankerProtocol


def safe_lookup(sample, var_name: str) -> int:
  # dimod SampleView.get() raises ValueError on unknown vars (Mapping.get
  # only catches KeyError), so 다항식 합성 중 계수가 0 으로 사라져 BQM 에
  # 등록되지 않은 변수는 default 0 으로 처리한다.
  try:
    return int(sample[var_name])
  except (KeyError, ValueError):
    return 0


def directed_from_bit(edge: Edge, bit: int) -> Edge:
  """무방향 간선 + 결정 비트 → 방향이 정해진 directed Edge (id 보존).

  bit==1 이면 max→min(vertices 역방향), 아니면 min→max. id 를 보존하므로
  Solution.edges dict 키가 원본 간선 id 와 일치한다(평행간선도 분리 유지).
  """
  if bit == 1:
    return Edge(edge.vertices[1], edge.vertices[0], edge.weight, True, id=edge.id)
  return Edge(edge.vertices[0], edge.vertices[1], edge.weight, True, id=edge.id)


def process_solution(
    best_sample: dict[str, int], canonical_edges: list[Edge]
) -> dict[int, Edge]:
  """best sample 를 edge_id → directed Edge dict 로 변환.

  Returns:
      dict[int, Edge]: 키는 Edge.id, 값은 방향이 정해진 directed Edge. id 키잉이라
      같은 양끝점을 잇는 평행 same-direction 간선도 붕괴하지 않고 따로 보존된다.
  """
  final_edges: dict[int, Edge] = {}
  for edge in canonical_edges:

    # handle predefined edges
    if edge.directed:
      final_edges[edge.id] = edge
      continue

    # handle optimized edges
    bit = safe_lookup(best_sample, edge.to_key())
    final_edges[edge.id] = directed_from_bit(edge, bit)

  return final_edges


def select_best_sample(
    sample_set: SampleSet,
    canonical_edges: list[Edge],
    ranker: SolutionRankerProtocol,
) -> dict[int, Edge]:
  if len(sample_set) == 0:
    return process_solution({}, canonical_edges)

  def get_effective_score(directed_edges: dict[int, Edge]):
    solution = Solution(
      edges=directed_edges,
      graph=Graph(edges=canonical_edges),
      sample_set=sample_set,
      score=None,
    )

    return ranker.run(solution)

  return min(
    map(lambda sample: process_solution(sample, canonical_edges), sample_set.samples()),
    key=get_effective_score
  )
