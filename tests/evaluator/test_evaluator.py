import pytest
from dimod import SampleSet

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import Evaluator


def _build_solution(
    edges: list[Edge],
    directed_edges: set[tuple[int, int]],
    samples=None,
    energies=None,
    num_occurrences=None,
) -> Solution:
  graph = Graph(edges=edges)
  # 샘플은 양끝점 키(예: (1, 2)) 로 비트를 주면 실제 Edge.to_key()(=e_{id}) 로 변환한다.
  key_by_endpoint = {e.endpoint_key(): e.to_key() for e in graph.edges.values()}
  weight_by_endpoint = {e.endpoint_key(): e.weight for e in graph.edges.values()}
  translated = (
    None
    if samples is None
    else [
      {key_by_endpoint[frozenset(endpoint)]: bit for endpoint, bit in sample.items()}
      for sample in samples
    ]
  )
  # Solution.edges 는 edge_id → directed Edge dict. directed_edges 튜플에 그래프
  # 가중치를 붙여 directed Edge 로 만든다.
  edges_dict = {
    index: Edge(source, target, weight_by_endpoint[frozenset({source, target})], True)
    for index, (source, target) in enumerate(directed_edges)
  }
  return Solution(
    edges=edges_dict,
    graph=graph,
    sample_set=SampleSet.from_samples(
      [] if translated is None else translated,
      vartype="BINARY",
      energy=[] if energies is None else energies,
      num_occurrences=num_occurrences,
    ),
    score=None,
  )


def test_eval_flow_returns_sum_of_squared_in_out_weight_differences() -> None:
  evaluator = Evaluator()

  solution = _build_solution(
    edges=[
      Edge(1, 2, 3, True),
      Edge(2, 3, 1, True),
      Edge(3, 1, 2, True),
    ],
    directed_edges={(1, 2), (2, 3), (3, 1)},
  )

  assert evaluator.eval_flow(solution) == pytest.approx(6.0)


def test_run_returns_score_for_solution_object() -> None:
  evaluator = Evaluator()

  score = evaluator.run(_build_solution(
    edges=[
      Edge(1, 2, 1, False),
      Edge(2, 3, 1, False),
      Edge(1, 3, 1, False),
    ],
    directed_edges={(1, 2), (2, 3), (3, 1)},
    samples=[{(1, 2): 0, (2, 3): 0, (1, 3): 1}],
    energies=[0.0],
  ))

  assert score.apsp_sum == 9.0
  assert score.strong_connect_rate == 1.0
  assert score.flow_score == 0.0
  assert score.sample_score == 0.0


def test_eval_strong_connect_rate_returns_fraction_of_strongly_connected_samples() -> None:
  evaluator = Evaluator()

  solution = _build_solution(
    edges=[
      Edge(1, 2, 1, False),
      Edge(2, 3, 1, False),
      Edge(1, 3, 1, False),
    ],
    directed_edges={(1, 2), (2, 3), (3, 1)},
    samples=[
      {(1, 2): 0, (2, 3): 0, (1, 3): 1},
      {(1, 2): 0, (2, 3): 1, (1, 3): 0},
    ],
    energies=[0.0, 1.0],
  )

  assert evaluator.eval_strong_connect_rate(solution) == pytest.approx(0.5)


def test_eval_strong_connect_rate_counts_sample_occurrences() -> None:
  evaluator = Evaluator()

  solution = _build_solution(
    edges=[
      Edge(1, 2, 1, False),
      Edge(2, 3, 1, False),
      Edge(1, 3, 1, False),
    ],
    directed_edges={(1, 2), (2, 3), (3, 1)},
    samples=[
      {(1, 2): 0, (2, 3): 0, (1, 3): 1},
      {(1, 2): 0, (2, 3): 1, (1, 3): 0},
    ],
    energies=[0.0, 1.0],
    num_occurrences=[2, 1],
  )

  assert evaluator.eval_strong_connect_rate(solution) == pytest.approx(2 / 3)


def test_eval_sample_score_returns_minimum_energy() -> None:
  evaluator = Evaluator()
  solution = _build_solution(
    edges=[Edge(1, 2, 1, False)],
    directed_edges={(1, 2)},
    samples=[{(1, 2): 0}, {(1, 2): 1}],
    energies=[3.5, -2.0],
  )

  assert evaluator.eval_sample_score(solution) == -2.0
