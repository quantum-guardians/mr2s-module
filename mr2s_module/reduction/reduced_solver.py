"""축약 presolve 를 감싼 솔버 래퍼 (ISSUE-52 간선 축약 3단계).

contract → inner solve → lift. inner solver 는 축약 그래프(전부 클론)만 보므로
원본 그래프는 in-place 방향 고정으로부터 안전하다. 반환 Solution 은 원본 그래프
기준이다: edges 는 원본 edge id 전량의 방향, graph 는 호출 시 받은 원본 객체.

inner solver 가 계산한 score 는 축약 그래프(harmonic super edge 가중치,
사이클 제거) 기준이라 원본 그래프의 점수가 아니므로 버린다. evaluator 를
주입하면 lift 된 Solution 을 원본 그래프 기준으로 재평가해 score 를 채우고,
주입하지 않으면 score=None 으로 두어 caller 가 직접 평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import EvaluatorProtocol, Mr2sSolverProtocol
from mr2s_module.reduction.chain_contraction import (
    SuperEdgeWeight,
    contract_chains,
    lift_solution_edges,
)
from mr2s_module.util.sample_set import empty_binary_sample_set


@dataclass
class ReductionMr2sSolver:
    """degree-2 체인 축약 presolve 래퍼. 임의의 Mr2sSolverProtocol inner 를 감싼다."""

    mr2s_solver: Mr2sSolverProtocol
    min_internal_vertices: int = 1
    super_edge_weight: SuperEdgeWeight = SuperEdgeWeight.HARMONIC
    evaluator: EvaluatorProtocol | None = None

    def run(self, graph: Graph) -> Solution:
        result = contract_chains(
            graph,
            min_internal_vertices=self.min_internal_vertices,
            super_edge_weight=self.super_edge_weight,
        )

        if result.contracted_graph.is_empty():
            # 그래프 전체가 사이클(들)로 소진된 퇴화 케이스: 변수가 0개이므로
            # solver 를 부르지 않고 lift 만으로 균일 배향을 만든다.
            lifted = lift_solution_edges({}, result)
            sample_set = empty_binary_sample_set()
        else:
            inner_solution = self.mr2s_solver.run(result.contracted_graph)
            lifted = lift_solution_edges(inner_solution.edges, result)
            # 축약이 실제로 일어났으면 inner sample 의 변수(super edge)는 원본 체인 간선을
            # 못 덮는다. 그 sample 을 물려주면 Evaluator 가 lifted 배향 대신 기본 정방향을
            # 채점하므로, 축약이 있었으면 sample 을 버리고 edges 기준으로 채점하게 둔다.
            sample_set = (
                empty_binary_sample_set()
                if result.has_chains
                else inner_solution.sample_set
            )

        solution = Solution(
            edges=lifted,
            graph=graph,
            sample_set=sample_set,
        )
        if self.evaluator is not None:
            solution.score = self.evaluator.run(solution)
        return solution
