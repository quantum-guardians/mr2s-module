from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast

import networkx as nx

from mr2s_module.domain import Score, Solution
from mr2s_module.evaluator.apsp_sum_ranker import ApspSumRanker
from mr2s_module.qubo.solution_processing import process_solution
from mr2s_module.util import flow_imbalance


class _SampleDatum(Protocol):
    """dimod SampleSet.data(fields=...) 의 행 — 스텁이 필드 namedtuple 을 표현하지
    못해 cast 대상으로만 쓰인다 (stub limitation)."""

    sample: Mapping[Any, int]
    num_occurrences: int


class Evaluator:
    def __init__(self):
        # 인스턴스로 들고 있어야 무방향 APSP 캐시가 solution 간에 재사용된다.
        self._apsp_ranker = ApspSumRanker()

    @staticmethod
    def _build_graph_from_edges(
        directed_edges: set[tuple[int, int]],
        vertices: set[int],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_edges_from(directed_edges)
        graph.add_nodes_from(vertices)
        return graph

    @staticmethod
    def _sample_to_directed_edges(
        sample,
        solution: Solution,
    ) -> set[tuple[int, int]]:
        # sample → 방향 변환은 QUBO 솔버와 같은 코어를 쓴다. 두 벌로 두면 분기한다.
        return set(
            process_solution(sample, list(solution.graph.edges.values())).values()
        )

    @staticmethod
    def _solution_to_directed_edges(solution: Solution) -> set[tuple[int, int]]:
        return set(solution.edges.values())

    @staticmethod
    def _is_strongly_connected(
        directed_edges: set[tuple[int, int]],
        vertices: set[int],
    ) -> bool:
        graph = Evaluator._build_graph_from_edges(directed_edges, vertices)
        return nx.is_strongly_connected(graph)

    def eval_apsp_sum(self, solution: Solution) -> float:
        """기본 method(stretch) 값. Score.apsp_sum 은 평균 스트레치 의미."""
        return self._apsp_ranker.run(solution)

    def eval_strong_connect_rate(self, solution: Solution) -> float:
        vertices = solution.graph.get_vertices()

        if not vertices:
            return 0.0

        if len(solution.sample_set) == 0:
            return float(
                self._is_strongly_connected(
                    self._solution_to_directed_edges(solution),
                    vertices,
                )
            )

        total_samples = 0
        strongly_connected_samples = 0

        data_rows = cast(
            "Iterable[_SampleDatum]",
            solution.sample_set.data(["sample", "num_occurrences"]),
        )
        for datum in data_rows:
            directed_edges = self._sample_to_directed_edges(datum.sample, solution)
            occurrences = int(datum.num_occurrences)

            total_samples += occurrences
            if self._is_strongly_connected(directed_edges, vertices):
                strongly_connected_samples += occurrences

        if total_samples == 0:
            return float(
                self._is_strongly_connected(
                    self._solution_to_directed_edges(solution),
                    vertices,
                )
            )

        return strongly_connected_samples / total_samples

    def eval_flow(self, solution: Solution) -> float:
        edge_weights = {
            edge.id: float(edge.weight) for edge in solution.graph.edges.values()
        }
        return flow_imbalance(
            (source, target, edge_weights[edge_id])
            for edge_id, (source, target) in solution.edges.items()
        )

    @staticmethod
    def eval_sample_score(solution: Solution) -> float:
        if len(solution.sample_set) == 0:
            return float("inf")
        return float(solution.sample_set.record.energy.min())

    def run(self, solution: Solution) -> Score:
        return Score(
            apsp_sum=self.eval_apsp_sum(solution),
            strong_connect_rate=self.eval_strong_connect_rate(solution),
            flow_score=self.eval_flow(solution),
            sample_score=self.eval_sample_score(solution),
        )
