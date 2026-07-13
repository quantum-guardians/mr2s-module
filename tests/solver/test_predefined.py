from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver
from mr2s_module.solver.predefined import (
  create_robbin_solver,
  create_ils_solver,
)

import pytest
import numpy as np
import random

from mr2s_module.domain import Edge, Graph
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.predefined import (
  create_dnc_sa_solver,
  create_dnc_qubo_solver,
  create_qubo_solver,
  create_sa_solver,
  create_qubo_sa_solver,
  create_qubo_qa_solver,
  create_dnc_qubo_sa_solver,
  create_dnc_qubo_qa_solver,
)
from tests.util.graph_fixtures import delaunay_graph


def test_create_robbin_solver() -> None:
  solver = create_robbin_solver()
  assert isinstance(solver, RobbinMR2SSolver)


def test_create_ils_solver() -> None:
  solver = create_ils_solver(max_iter=5)
  assert isinstance(solver, IlsMR2SSolver)


def test_create_sa_solver() -> None:
  solver = create_sa_solver(random_seed=42)
  assert solver.random_seed == 42
  assert solver.apsp_weight == 1.0


def test_create_qubo_solver() -> None:
  solver = create_qubo_solver()
  assert solver.qubo_solver is not None


def test_create_qubo_sa_solver() -> None:
  solver = create_qubo_sa_solver()
  assert solver.qubo_solver is not None


def test_create_qubo_qa_solver(monkeypatch) -> None:
  class FakeDWaveSampler:
    pass

  class FakeEmbeddingComposite:
    def __init__(self, child):
      self.child = child

  import mr2s_module.qubo.qubo_solver as qubo_solver_module
  monkeypatch.setattr(qubo_solver_module, "DWaveSampler", FakeDWaveSampler)
  monkeypatch.setattr(qubo_solver_module, "EmbeddingComposite", FakeEmbeddingComposite)

  solver = create_qubo_qa_solver()
  assert solver.qubo_solver is not None


def test_create_dnc_sa_solver() -> None:
  solver = create_dnc_sa_solver(max_vertices=50, random_seed=42)
  assert solver.graph_partition_strategy is not None
  assert solver.graph_partition_strategy.max_vertices == 50


def test_create_dnc_qubo_solver() -> None:
  solver = create_dnc_qubo_solver()
  assert solver.graph_partition_strategy is not None


def test_create_dnc_qubo_sa_solver() -> None:
  solver = create_dnc_qubo_sa_solver()
  assert solver.graph_partition_strategy is not None


def test_create_dnc_qubo_qa_solver(monkeypatch) -> None:
  class FakeDWaveSampler:
    pass

  class FakeEmbeddingComposite:
    def __init__(self, child):
      self.child = child

  import mr2s_module.qubo.qubo_solver as qubo_solver_module
  monkeypatch.setattr(qubo_solver_module, "DWaveSampler", FakeDWaveSampler)
  monkeypatch.setattr(qubo_solver_module, "EmbeddingComposite", FakeEmbeddingComposite)

  solver = create_dnc_qubo_qa_solver()
  assert solver.graph_partition_strategy is not None


def test_vertex_count_partition_strategy_small_graph() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  face_cycle = FaceClusterPartition(
    target_k=2,
    clusterer=KMeansFaceClusterer(),
  )
  strategy = VertexCountPartitionStrategy(face_cycle=face_cycle, max_vertices=5)
  partition = strategy.run(graph)

  assert len(partition.sub_graphs) == 1
  assert partition.sub_graphs[0] is graph


def test_vertex_count_partition_strategy_divides_when_exceeds_max_vertices() -> None:
  # Capture state
  np_state = np.random.get_state()
  py_state = random.getstate()
  try:
    np.random.seed(42)
    random.seed(42)
    graph = delaunay_graph(n=20, seed=42)

    face_cycle = FaceClusterPartition(
      target_k=2,
      clusterer=KMeansFaceClusterer(),
    )

    # Setting max_vertices=15 so that the entire graph (20 vertices) must be divided
    strategy = VertexCountPartitionStrategy(face_cycle=face_cycle, max_vertices=15)
    partition = strategy.run(graph)

    assert len(partition.sub_graphs) > 1
    for sub_graph in partition.sub_graphs:
      assert len(sub_graph.get_vertices()) <= 15
  finally:
    # Restore state
    np.random.set_state(np_state)
    random.setstate(py_state)


def test_dnc_sa_solver_runs_end_to_end() -> None:
  # Capture state
  np_state = np.random.get_state()
  py_state = random.getstate()
  try:
    np.random.seed(42)
    random.seed(42)
    graph = delaunay_graph(n=20, seed=42)

    # A solver with max_vertices=15 will force division of the graph
    solver = create_dnc_sa_solver(max_vertices=15, random_seed=42)
    solution = solver.run(graph)

    assert len(solution.edges) == len(graph.edges)
    assert solution.score is not None
  finally:
    # Restore state
    np.random.set_state(np_state)
    random.setstate(py_state)
