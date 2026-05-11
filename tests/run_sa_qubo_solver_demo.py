from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import (
  ApspSumRanker,
  Edge,
  Evaluator,
  FaceCycle,
  FlowPolyGenerator,
  Graph,
  NHop,
  NHopPolyGenerator,
  SAQuboSolver,
  SmallWorldSpec,
)
from mr2s_module.domain import Solution
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm


def build_planar_graph(num_points: int, seed: int, weight: int = 1) -> Graph:
  rng = np.random.default_rng(seed)
  points = rng.random((num_points, 2))
  tri = Delaunay(points)

  seen: set[tuple[int, int]] = set()
  edges: list[Edge] = []
  for simplex in tri.simplices:
    for u, v in itertools.combinations(simplex, 2):
      u, v = int(u), int(v)
      if u == v:
        continue

      edge_id = (min(u, v), max(u, v))
      if edge_id in seen:
        continue

      seen.add(edge_id)
      edges.append(Edge(edge_id[0], edge_id[1], weight, False))

  return Graph(edges=edges)


def thin_planar_graph(
    graph: Graph,
    seed: int,
    remove_ratio: float,
    weight: int = 1,
) -> Graph:
  if not 0.0 <= remove_ratio < 1.0:
    raise ValueError(f"remove_ratio must be in [0.0, 1.0), got {remove_ratio}")

  nx_graph = nx.Graph()
  nx_graph.add_edges_from(edge.id for edge in graph.edges)

  rng = np.random.default_rng(seed)
  edges = list(nx_graph.edges())
  rng.shuffle(edges)

  keep_ratio = 1.0 - remove_ratio
  target_edge_count = int(len(edges) * keep_ratio)

  for u, v in edges:
    if nx_graph.number_of_edges() <= target_edge_count:
      break

    nx_graph.remove_edge(u, v)
    if not nx.is_biconnected(nx_graph):
      nx_graph.add_edge(u, v)

  nx_graph.remove_nodes_from([vertex for vertex in nx_graph.nodes() if nx_graph.degree(vertex) == 0])

  return Graph(edges=[
    Edge(min(u, v), max(u, v), weight, False)
    for u, v in nx_graph.edges()
  ])


class ConfiguredSAQuboSolver(SAQuboSolver):
  def __init__(self, ranker, num_reads: int):
    super().__init__(ranker)
    self.num_reads = num_reads

  def run(self, qubo, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo, num_reads=self.num_reads)
    return Solution(
      edges=self._select_best_sample(sample_set, graph.edges),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )


def build_polynomial(graph: Graph):
  n_hop_generator = NHopPolyGenerator()
  n_hop_generator.small_world_spec = SmallWorldSpec(
    n_hops=[NHop(n=2, weight=1)]
  )

  polynomial = FlowPolyGenerator().run(graph)
  return add_polys(polynomial, n_hop_generator.run(graph))


def describe_result(graph: Graph, score, solution: Solution) -> None:
  print("Graph")
  print(f"  vertices: {len(graph.get_vertices())}")
  print(f"  edges: {len(graph.edges)}")
  print()

  print("Score")
  print(f"  apsp_sum: {score.apsp_sum}")
  print(f"  strong_connect_rate: {score.strong_connect_rate:.4f}")
  print(f"  flow_score: {score.flow_score}")
  print(f"  sample_score: {score.sample_score}")
  print()

  print("Selected orientation")
  sorted_edges = sorted(solution.edges)
  preview = sorted_edges[: min(20, len(sorted_edges))]
  print(f"  directed_edges_preview({len(preview)}): {preview}")
  if len(sorted_edges) > len(preview):
    print(f"  ... {len(sorted_edges) - len(preview)} more edges")
  print()

  print("Sample set")
  print(f"  sample_count: {len(solution.sample_set)}")
  print(f"  total_occurrences: {int(solution.sample_set.record.num_occurrences.sum())}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run SA-QUBO solver on a generated planar graph."
  )
  # Number of points used to generate the planar Delaunay graph.
  parser.add_argument("--num-points", type=int, default=20)
  # Random seed for reproducible planar graph generation.
  parser.add_argument("--seed", type=int, default=7)
  # Weight assigned uniformly to every generated edge.
  parser.add_argument("--weight", type=int, default=1)
  # Fraction of edges to remove while trying to keep the graph biconnected.
  parser.add_argument("--remove-ratio", type=float, default=0.0)
  # Number of simulated annealing reads (samples) to draw.
  parser.add_argument("--num-reads", type=int, default=100)
  # FaceCycle target k used when preprocessing is enabled.
  parser.add_argument("--target-k", type=int, default=8)
  # Apply FaceCycle preprocessing before building the QUBO.
  parser.add_argument("--use-face-cycle", action="store_true")
  args = parser.parse_args()

  graph = build_planar_graph(
    num_points=args.num_points,
    seed=args.seed,
    weight=args.weight,
  )
  original_edge_count = len(graph.edges)

  if args.remove_ratio > 0.0:
    graph = thin_planar_graph(
      graph=graph,
      seed=args.seed,
      remove_ratio=args.remove_ratio,
      weight=args.weight,
    )

  if args.use_face_cycle:
    predefined_edges = FaceCycle(target_k=args.target_k).run(graph)
    graph.define_edge_direction(predefined_edges)

  polynomial = build_polynomial(graph)
  bqm = map_binary_poly_to_bqm(polynomial)
  ranking_solver = ConfiguredSAQuboSolver(
    ranker=ApspSumRanker(),
    num_reads=args.num_reads,
  )
  solution = ranking_solver.run(bqm, graph)
  score = Evaluator().run(solution)

  nx_graph = nx.DiGraph()
  nx_graph.add_nodes_from(graph.get_vertices())
  nx_graph.add_edges_from(solution.edges)
  print(f"Strongly connected: {nx.is_strongly_connected(nx_graph)}")
  print(f"Original edge count: {original_edge_count}")
  print(f"Final edge count: {len(graph.edges)}")
  print()
  describe_result(graph, score, solution)


if __name__ == "__main__":
  main()
