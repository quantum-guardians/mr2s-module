from __future__ import annotations

from typing import Any

from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol
from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.qubo import QuboSolver
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.partition.degeneracy_pruning import (
  DegeneracyPruningFaceCyclePartitionStrategy,
)


def create_robbin_solver(
    evaluator: EvaluatorProtocol = Evaluator(),
) -> RobbinMR2SSolver:
  """Creates a standalone Robbin MR2S solver."""
  return RobbinMR2SSolver(evaluator=evaluator)


def create_ils_solver(
    max_iter: int = 30,
    patience: int = 5,
    is_relaxed: bool = False,
    perturb_strength: int = 2,
    evaluator: EvaluatorProtocol = Evaluator(),
) -> IlsMR2SSolver:
  """Creates a standalone Iterated Local Search MR2S solver."""
  return IlsMR2SSolver(
    max_iter=max_iter,
    patience=patience,
    is_relaxed=is_relaxed,
    perturb_strength=perturb_strength,
    evaluator=evaluator,
  )


def create_sa_solver(
    *,
    sweeps_per_temperature: int = 2,
    num_restarts: int = 4,
    random_seed: int | None = None,
    apsp_weight: float = 1.0,
    flow_weight: float = 1.0,
    disconnected_pair_penalty: float = 10.0,
) -> SAMR2SSolver:
  """Creates a standard Simulated Annealing solver."""
  return SAMR2SSolver(
    sweeps_per_temperature=sweeps_per_temperature,
    num_restarts=num_restarts,
    random_seed=random_seed,
    apsp_weight=apsp_weight,
    flow_weight=flow_weight,
    disconnected_pair_penalty=disconnected_pair_penalty,
  )


def create_qubo_sa_solver() -> QuboMR2SSolver:
  """Creates a standard QUBO MR2S solver using Simulated Annealing backend."""
  return QuboMR2SSolver(
    qubo_solver=QuboSolver.create_sa_solver(ranker=ApspSumRanker())
  )


def create_qubo_qa_solver() -> QuboMR2SSolver:
  """Creates a QUBO MR2S solver using D-Wave Quantum Annealing Hardware backend."""
  return QuboMR2SSolver(
    qubo_solver=QuboSolver.create_qa_solver(ranker=ApspSumRanker())
  )


def create_qubo_solver() -> QuboMR2SSolver:
  """Creates a standard QUBO MR2S solver."""
  return create_qubo_sa_solver()


def create_dnc_sa_solver(
    max_vertices: int = 100,
    *,
    sweeps_per_temperature: int = 2,
    num_restarts: int = 4,
    random_seed: int | None = None,
    apsp_weight: float = 1.0,
    flow_weight: float = 1.0,
    disconnected_pair_penalty: float = 10.0,
) -> DnCMr2sSolver:
  """Creates a DnC solver that divides the graph to have <= max_vertices vertices

  and solves subgraphs using Simulated Annealing.
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
  return DnCMr2sSolver(
    mr2s_solver=sa_solver,
    face_cycle=face_cycle,
    graph_partition_strategy=partition_strategy,
  )


def create_dnc_qubo_sa_solver(
    target_graph: Any = None,
) -> DnCMr2sSolver:
  """Creates a DnC solver that divides the graph and solves subgraphs using QUBO SA."""
  qubo_solver = create_qubo_sa_solver()
  face_cycle = FaceClusterPartition(
    target_k=2,
    clusterer=KMeansFaceClusterer(),
  )
  partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    target_graph=target_graph,
  )
  return DnCMr2sSolver(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    graph_partition_strategy=partition_strategy,
    target_graph=target_graph,
  )


def create_dnc_qubo_qa_solver(
    target_graph: Any = None,
) -> DnCMr2sSolver:
  """Creates a DnC solver that divides the graph and solves subgraphs using QUBO QA."""
  qubo_solver = create_qubo_qa_solver()
  face_cycle = FaceClusterPartition(
    target_k=2,
    clusterer=KMeansFaceClusterer(),
  )
  partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    target_graph=target_graph,
  )
  return DnCMr2sSolver(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    graph_partition_strategy=partition_strategy,
    target_graph=target_graph,
  )


def create_dnc_qubo_solver(
    target_graph: Any = None,
) -> DnCMr2sSolver:
  """Creates a DnC solver that divides the graph

  and solves subgraphs using QUBO.
  """
  return create_dnc_qubo_sa_solver(target_graph=target_graph)
