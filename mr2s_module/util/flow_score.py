from typing import Iterable


def flow_imbalance(
    directed_weighted_edges: Iterable[tuple[int, int, float]],
) -> float:
  """flow 균형 점수의 단일 수치 진실 공급원: Σ_v (Σ_in w − Σ_out w)².

  (source, target, weight) 튜플 단위로 받으므로 그래프/간선 정체성과 무관하다.
  평행 간선은 튜플을 여러 번 넘기면 그대로 합산된다. 간선이 하나도 닿지 않는
  정점은 기여가 0이라 정점 목록을 따로 받을 필요가 없다.

  FlowPolyGenerator 의 다항식과 같은 수식이어야 한다 —
  tests/qubo/test_flow_poly_equivalence.py 가 동치성을 고정한다.
  """
  balance: dict[int, float] = {}
  for source, target, weight in directed_weighted_edges:
    balance[source] = balance.get(source, 0.0) - weight
    balance[target] = balance.get(target, 0.0) + weight
  return float(sum(value * value for value in balance.values()))
