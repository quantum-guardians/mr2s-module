from __future__ import annotations

from typing import TypeVar

import networkx as nx

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.evaluator import ApspSumRanker, Evaluator
from mr2s_module.protocols import EvaluatorProtocol, Mr2sSolverProtocol
from mr2s_module.qubo import QuboSolver
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver
from mr2s_module.solver.partition.degeneracy_pruning import (
    DegeneracyPruningFaceCyclePartitionStrategy,
)
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver

_SolverT = TypeVar("_SolverT", bound=Mr2sSolverProtocol)


def _wrap_reduction(
    solver: _SolverT,
    use_reduction: bool,
    evaluator: EvaluatorProtocol,
) -> _SolverT | ReductionMr2sSolver:
    """Wraps a solver with degree-2 chain contraction presolve when enabled.

    The wrapper contracts degree-2 chains into super edges (and removes hanging
    cycles) before solving, then lifts the directions back onto the original
    edges. The lifted Solution is re-scored on the original graph with the given
    evaluator.
    """
    if not use_reduction:
        return solver
    return ReductionMr2sSolver(mr2s_solver=solver, evaluator=evaluator)


def create_robbin_solver(
    evaluator: EvaluatorProtocol = Evaluator(),
    use_reduction: bool = True,
) -> RobbinMR2SSolver | ReductionMr2sSolver:
    """Creates a Robbin MR2S solver with chain contraction presolve by default."""
    return _wrap_reduction(
        RobbinMR2SSolver(evaluator=evaluator), use_reduction, evaluator
    )


def create_ils_solver(
    max_iter: int = 30,
    patience: int = 5,
    is_relaxed: bool = False,
    perturb_strength: int = 2,
    evaluator: EvaluatorProtocol = Evaluator(),
    use_reduction: bool = True,
) -> IlsMR2SSolver | ReductionMr2sSolver:
    """Creates an Iterated Local Search MR2S solver with chain contraction presolve by default."""
    solver = IlsMR2SSolver(
        max_iter=max_iter,
        patience=patience,
        is_relaxed=is_relaxed,
        perturb_strength=perturb_strength,
        evaluator=evaluator,
    )
    return _wrap_reduction(solver, use_reduction, evaluator)


def create_sa_solver(
    *,
    sweeps_per_temperature: int = 2,
    num_restarts: int = 4,
    random_seed: int | None = None,
    apsp_weight: float = 1.0,
    flow_weight: float = 1.0,
    disconnected_pair_penalty: float = 10.0,
    use_reduction: bool = True,
) -> SAMR2SSolver | ReductionMr2sSolver:
    """Creates a Simulated Annealing solver with chain contraction presolve by default."""
    solver = SAMR2SSolver(
        sweeps_per_temperature=sweeps_per_temperature,
        num_restarts=num_restarts,
        random_seed=random_seed,
        apsp_weight=apsp_weight,
        flow_weight=flow_weight,
        disconnected_pair_penalty=disconnected_pair_penalty,
    )
    return _wrap_reduction(solver, use_reduction, Evaluator())


def _build_qubo_sa_solver() -> QuboMR2SSolver:
    return QuboMR2SSolver(
        qubo_solver=QuboSolver.create_sa_solver(ranker=ApspSumRanker())
    )


def _build_qubo_qa_solver() -> QuboMR2SSolver:
    return QuboMR2SSolver(
        qubo_solver=QuboSolver.create_qa_solver(ranker=ApspSumRanker())
    )


def create_qubo_sa_solver(
    use_reduction: bool = True,
) -> QuboMR2SSolver | ReductionMr2sSolver:
    """Creates a QUBO MR2S solver (SA backend) with chain contraction presolve by default."""
    return _wrap_reduction(_build_qubo_sa_solver(), use_reduction, Evaluator())


def create_qubo_qa_solver(
    use_reduction: bool = True,
) -> QuboMR2SSolver | ReductionMr2sSolver:
    """Creates a QUBO MR2S solver (D-Wave QA backend) with chain contraction presolve by default."""
    return _wrap_reduction(_build_qubo_qa_solver(), use_reduction, Evaluator())


def create_qubo_solver(
    use_reduction: bool = True,
) -> QuboMR2SSolver | ReductionMr2sSolver:
    """Creates a standard QUBO MR2S solver."""
    return create_qubo_sa_solver(use_reduction=use_reduction)


def create_dnc_sa_solver(
    max_vertices: int = 100,
    *,
    sweeps_per_temperature: int = 2,
    num_restarts: int = 4,
    random_seed: int | None = None,
    apsp_weight: float = 1.0,
    flow_weight: float = 1.0,
    disconnected_pair_penalty: float = 10.0,
    use_reduction: bool = True,
) -> DnCMr2sSolver | ReductionMr2sSolver:
    """Creates a DnC solver that divides the graph to have <= max_vertices vertices

    and solves subgraphs using Simulated Annealing. Chain contraction presolve is
    applied to the whole graph before partitioning by default.
    """
    sa_solver = SAMR2SSolver(
        sweeps_per_temperature=sweeps_per_temperature,
        num_restarts=num_restarts,
        random_seed=random_seed,
        apsp_weight=apsp_weight,
        flow_weight=flow_weight,
        disconnected_pair_penalty=disconnected_pair_penalty,
    )
    face_cycle = FaceClusterPartition(
        target_k=2,
        clusterer=KMeansFaceClusterer(),
    )
    partition_strategy = VertexCountPartitionStrategy(
        face_cycle=face_cycle,
        max_vertices=max_vertices,
    )
    solver = DnCMr2sSolver(
        mr2s_solver=sa_solver,
        face_cycle=face_cycle,
        graph_partition_strategy=partition_strategy,
    )
    return _wrap_reduction(solver, use_reduction, Evaluator())


def create_dnc_qubo_sa_solver(
    target_graph: nx.Graph | None = None,
    use_reduction: bool = True,
) -> DnCMr2sSolver | ReductionMr2sSolver:
    """Creates a DnC solver that divides the graph and solves subgraphs using QUBO SA.

    Chain contraction presolve is applied by default; the partition strategy then
    sees the contracted graph, which is intended — embedding estimates against
    target_graph (the QPU topology) are computed for the graph actually solved.
    """
    qubo_solver = _build_qubo_sa_solver()
    face_cycle = FaceClusterPartition(
        target_k=2,
        clusterer=KMeansFaceClusterer(),
    )
    partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        target_graph=target_graph,
    )
    solver = DnCMr2sSolver(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        graph_partition_strategy=partition_strategy,
        target_graph=target_graph,
    )
    return _wrap_reduction(solver, use_reduction, Evaluator())


def create_dnc_qubo_qa_solver(
    target_graph: nx.Graph | None = None,
    use_reduction: bool = True,
) -> DnCMr2sSolver | ReductionMr2sSolver:
    """Creates a DnC solver that divides the graph and solves subgraphs using QUBO QA.

    Chain contraction presolve is applied by default; the partition strategy then
    sees the contracted graph, which is intended — embedding estimates against
    target_graph (the QPU topology) are computed for the graph actually solved.
    """
    qubo_solver = _build_qubo_qa_solver()
    face_cycle = FaceClusterPartition(
        target_k=2,
        clusterer=KMeansFaceClusterer(),
    )
    partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        target_graph=target_graph,
    )
    solver = DnCMr2sSolver(
        mr2s_solver=qubo_solver,
        face_cycle=face_cycle,
        graph_partition_strategy=partition_strategy,
        target_graph=target_graph,
    )
    return _wrap_reduction(solver, use_reduction, Evaluator())


def create_dnc_qubo_solver(
    target_graph: nx.Graph | None = None,
    use_reduction: bool = True,
) -> DnCMr2sSolver | ReductionMr2sSolver:
    """Creates a DnC solver that divides the graph

    and solves subgraphs using QUBO.
    """
    return create_dnc_qubo_sa_solver(
        target_graph=target_graph,
        use_reduction=use_reduction,
    )
