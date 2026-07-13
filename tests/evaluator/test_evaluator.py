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
  # directed_edges(set[tuple]) → edge id 키 dict 로 매핑(끝점 매칭).
  edges_by_id: dict[int, tuple[int, int]] = {}
  # 구식 변수명(e_{u}_{v}) → 신식 id 기반 to_key(e_{id}) 변환 맵.
  key_map: dict[str, str] = {}
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    key_map[f"e_{u}_{v}"] = edge.to_key()
    if edge.directed and edge.vertices in directed_edges:
      edges_by_id[edge.id] = edge.vertices
    elif (u, v) in directed_edges:
      edges_by_id[edge.id] = (u, v)
    elif (v, u) in directed_edges:
      edges_by_id[edge.id] = (v, u)
  if samples is not None:
    samples = [
      {key_map.get(k, k): val for k, val in sample.items()}
      for sample in samples
    ]
  return Solution(
    edges=edges_by_id,
    graph=graph,
    sample_set=SampleSet.from_samples(
      [] if samples is None else samples,
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
    samples=[{"e_1_2": 0, "e_2_3": 0, "e_1_3": 1}],
    energies=[0.0],
  ))

  assert score.apsp_sum == 1.5
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
      {"e_1_2": 0, "e_2_3": 0, "e_1_3": 1},
      {"e_1_2": 0, "e_2_3": 1, "e_1_3": 0},
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
      {"e_1_2": 0, "e_2_3": 0, "e_1_3": 1},
      {"e_1_2": 0, "e_2_3": 1, "e_1_3": 0},
    ],
    energies=[0.0, 1.0],
    num_occurrences=[2, 1],
  )

  assert evaluator.eval_strong_connect_rate(solution) == pytest.approx(2 / 3)


def test_eval_strong_connect_rate_uses_solution_edges_when_samples_empty() -> None:
  evaluator = Evaluator()

  solution = _build_solution(
    edges=[
      Edge(1, 2, 1, True),
      Edge(2, 1, 1, True),
    ],
    directed_edges={(1, 2), (2, 1)},
  )

  assert evaluator.eval_strong_connect_rate(solution) == 1.0


def test_eval_sample_score_returns_minimum_energy() -> None:
  evaluator = Evaluator()
  solution = _build_solution(
    edges=[Edge(1, 2, 1, False)],
    directed_edges={(1, 2)},
    samples=[{"e_1_2": 0}, {"e_1_2": 1}],
    energies=[3.5, -2.0],
  )

  assert evaluator.eval_sample_score(solution) == -2.0
