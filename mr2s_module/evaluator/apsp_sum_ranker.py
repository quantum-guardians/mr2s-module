from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from mr2s_module.domain import Solution
from mr2s_module.evaluator.distance_util import (
    UndirectedApspCache,
    build_directed_distance_graph,
    stretch_totals,
)

ApspMethod = Literal["stretch", "efficiency", "sum"]

_METHODS = ("stretch", "efficiency", "sum")


@dataclass
class ApspSumRanker:
    """APSP 기반 방향화 품질 ranker. 간선 거리 = 1/weight, 모든 method 낮을수록 좋음.

    - "stretch" (기본): 쌍별 (방향화 거리 / 무방향 거리) 평균. [1, ∞).
      그래프 크기·weight 스케일 불변. 비강연결이면 inf.
    - "efficiency": 1 − (E_방향화 / E_무방향), E = Σ 1/d (Latora–Marchiori).
      [0, 1], 0 = 무방향 거리 완전 보존. 도달 불가 쌍은 기여 0 이라 비강연결에도
      inf 없이 부분 점수 — 탐색 중간 해 평가용.
    - "sum": 방향화 APSP 거리 합. 비강연결이면 inf. 정규화 없음(같은 그래프
      안에서의 비교 전용).
    """

    method: ApspMethod = "stretch"
    _undirected_cache: UndirectedApspCache = field(
        init=False, repr=False, default_factory=UndirectedApspCache
    )

    def __post_init__(self):
        if self.method not in _METHODS:
            raise ValueError(
                f"Unknown method {self.method!r}, expected one of {_METHODS}"
            )

    # --- 진입점: 공용 APSP 계산 후 method 로 분기 ---

    def run(self, solution: Solution) -> float:
        directed = build_directed_distance_graph(solution)
        directed_lengths = dict(
            nx.all_pairs_dijkstra_path_length(directed, weight="distance")
        )
        vertices = solution.graph.get_vertices()

        if self.method == "efficiency":
            return self._efficiency(solution, directed_lengths, vertices)
        if not nx.is_strongly_connected(directed):
            return float("inf")
        if self.method == "stretch":
            return self._stretch(solution, directed_lengths, vertices)
        return self._sum(directed_lengths, vertices)

    # --- method 별 구현 ---

    def _stretch(
        self,
        solution: Solution,
        directed_lengths: dict[int, dict[int, float]],
        vertices: set[int],
    ) -> float:
        undirected_lengths = self._undirected_cache.get_lengths(solution.graph)

        # 강연결일 때만 호출되므로 unreachable=0, reachable=pair_count.
        total_stretch, pair_count, _ = stretch_totals(
            directed_lengths, undirected_lengths, vertices
        )
        if pair_count == 0:
            return 1.0
        return total_stretch / pair_count

    def _efficiency(
        self,
        solution: Solution,
        directed_lengths: dict[int, dict[int, float]],
        vertices: set[int],
    ) -> float:
        undirected_lengths = self._undirected_cache.get_lengths(solution.graph)

        directed_efficiency = 0.0
        undirected_efficiency = 0.0
        for source in vertices:
            for target in vertices:
                if source == target:
                    continue
                undirected_distance = undirected_lengths.get(source, {}).get(target)
                if undirected_distance is not None:
                    undirected_efficiency += 1.0 / undirected_distance
                directed_distance = directed_lengths.get(source, {}).get(target)
                if directed_distance is not None:
                    directed_efficiency += 1.0 / directed_distance

        # 쌍 수 정규화 항 n(n-1) 은 분자·분모에서 약분된다.
        if undirected_efficiency == 0.0:
            return 1.0
        return 1.0 - directed_efficiency / undirected_efficiency

    @staticmethod
    def _sum(
        directed_lengths: dict[int, dict[int, float]], vertices: set[int]
    ) -> float:
        total_distance = 0.0
        for source in vertices:
            for target in vertices:
                if source == target:
                    continue
                total_distance += directed_lengths[source][target]
        return total_distance
