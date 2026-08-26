"""hop 조합·축약 on/off 에 맞는 DnC + QUBO-SA 솔버 조립.

mr2s_module.solver.predefined.create_dnc_qubo_sa_solver 와 같은 구성이되, 실험에 필요한
세 가지를 추가한다: (1) hop 조합 주입, (2) seed 고정 C++ SA 샘플러와 num_reads,
(3) 축약 래퍼가 버리는 DnC 메타데이터(서브그래프 수, BQM 크기)를 기록하는 프록시.
tests/experiments/test_solvers.py 가 팩토리와의 구성 일치를 고정한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dimod
from dwave.samplers import SimulatedAnnealingSampler

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import ApspSumRanker, Evaluator
from mr2s_module.protocols import EvaluatorProtocol, Mr2sSolverProtocol
from mr2s_module.qubo import (
    FlowPolyGenerator,
    NHop,
    NHopPolyGenerator,
    QuboSolver,
    SmallWorldSpec,
)
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.degeneracy_pruning import (
    DegeneracyPruningFaceCyclePartitionStrategy,
)
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver

SAMPLER_NAME = "dwave.samplers.SimulatedAnnealingSampler"


def canonical_bqm(bqm: Any) -> dimod.BinaryQuadraticModel:
    """변수 순서를 라벨 정렬로 고정한 BQM 사본.

    FlowPolyGenerator 가 set[Edge] 를 순회하고 Edge 는 주소 해시를 쓰므로 BQM 의 변수
    삽입 순서가 프로세스마다 달라진다. SA 는 변수 순서에 따라 같은 seed 로도 다른
    샘플을 내므로, 순서를 정렬해 실행을 결정적으로 만든다. 라벨은 그대로라 결과
    SampleSet 은 원본 BQM 과 호환된다.
    """
    ordered = dimod.BinaryQuadraticModel(bqm.vartype)
    for variable in sorted(bqm.variables, key=str):
        ordered.add_variable(variable, bqm.get_linear(variable))
    for (u, v), bias in sorted(bqm.quadratic.items(), key=lambda item: str(item[0])):
        ordered.add_quadratic(u, v, bias)
    ordered.offset = bqm.offset
    return ordered


class SeededSampler:
    """sample() 호출마다 고정 seed 를 주입하고 변수 순서를 정규화하는 샘플러 어댑터."""

    def __init__(self, sampler: Any, seed: int) -> None:
        self._sampler = sampler
        self.seed = seed

    def sample(self, bqm: Any, **kwargs: Any) -> Any:
        return self._sampler.sample(canonical_bqm(bqm), seed=self.seed, **kwargs)


class RecordingSolver:
    """inner solver 의 마지막 Solution 을 보관하는 프록시 (Mr2sSolverProtocol 충족)."""

    def __init__(self, inner: Mr2sSolverProtocol) -> None:
        self.inner = inner
        self.last_solution: Solution | None = None
        self.last_graph: Graph | None = (
            None  # inner 가 실제로 푼 그래프 (축약 시 축약 그래프)
        )

    @property
    def evaluator(self) -> EvaluatorProtocol:
        return self.inner.evaluator

    def run(self, graph: Graph) -> Solution:
        self.last_graph = graph
        self.last_solution = self.inner.run(graph)
        return self.last_solution


def build_poly_generators(hops: Sequence[int]) -> list[Any]:
    if not hops:
        raise ValueError("hops must not be empty")
    return [
        FlowPolyGenerator(),
        NHopPolyGenerator(
            small_world_spec=SmallWorldSpec(n_hops=[NHop(n, 1) for n in hops])
        ),
    ]


def build_qubo_solver(
    hops: Sequence[int], *, seed: int, num_reads: int
) -> QuboMR2SSolver:
    sampler = SeededSampler(SimulatedAnnealingSampler(), seed)
    return QuboMR2SSolver(
        qubo_solver=QuboSolver(
            ranker=ApspSumRanker(), sampler=sampler, num_reads=num_reads
        ),
        evaluator=Evaluator(),
        poly_generators=build_poly_generators(hops),
    )


def build_solver(
    hops: Sequence[int],
    use_reduction: bool,
    *,
    seed: int,
    num_reads: int,
    use_dnc: bool = True,
) -> tuple[ReductionMr2sSolver | RecordingSolver, RecordingSolver]:
    """(최상위 솔버, inner 결과 기록 프록시).

    use_dnc=True 면 create_dnc_qubo_sa_solver 와 같은 DnC 구성, False 면 DnC 없이
    QuboMR2SSolver 가 그래프 전체를 한 QUBO 로 푼다 (QPU 제약이 없는 SA 대조군).
    """
    qubo_solver = build_qubo_solver(hops, seed=seed, num_reads=num_reads)
    if not use_dnc:
        recorder = RecordingSolver(qubo_solver)
        if not use_reduction:
            return recorder, recorder
        return ReductionMr2sSolver(
            mr2s_solver=recorder, evaluator=Evaluator()
        ), recorder
    face_cycle = FaceClusterPartition(target_k=2, clusterer=KMeansFaceClusterer())
    partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        target_graph=None,
    )
    dnc = DnCMr2sSolver(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        graph_partition_strategy=partition_strategy,
        target_graph=None,
    )
    recorder = RecordingSolver(dnc)
    if not use_reduction:
        return recorder, recorder
    return ReductionMr2sSolver(mr2s_solver=recorder, evaluator=Evaluator()), recorder
