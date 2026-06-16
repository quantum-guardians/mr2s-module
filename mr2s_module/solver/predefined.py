from typing import Any

from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.partition.degeneracy_pruning import (
  DegeneracyPruningFaceCyclePartitionStrategy,
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


def create_qubo_solver() -> QuboMR2SSolver:
  """Creates a standard QUBO MR2S solver."""
  return QuboMR2SSolver()


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
    max_vertices=max_vertices,
  )


def create_dnc_qubo_solver(
    max_vertices: int = 100,
    target_graph: Any = None,
) -> DnCMr2sSolver:
  """Creates a DnC solver that divides the graph to have <= max_vertices variables

  and solves subgraphs using QUBO.
  """
  qubo_solver = QuboMR2SSolver()
  face_cycle = FaceClusterPartition(
    target_k=2,
    clusterer=KMeansFaceClusterer(),
  )
  partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    target_graph=target_graph,
    max_vertices=max_vertices,
  )
  return DnCMr2sSolver(
    mr2s_solver=qubo_solver,
    face_cycle=face_cycle,
    graph_partition_strategy=partition_strategy,
    max_vertices=max_vertices,
    target_graph=target_graph,
  )
