from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import (
  SAMR2SSolver,
  Evaluator,
  Graph,
  Edge,
  FaceClusterPartition,
)
from mr2s_module.domain import Solution
from tests.util.graph_fixtures import delaunay_graph


def build_planar_graph(num_points: int, seed: int, weight: int = 1) -> Graph:
  return delaunay_graph(num_points, seed, weight=weight)


def thin_planar_graph(
    graph: Graph,
    seed: int,
    remove_ratio: float,
    weight: int = 1,
) -> Graph:
  if not 0.0 <= remove_ratio < 1.0:
    raise ValueError(f"remove_ratio must be in [0.0, 1.0), got {remove_ratio}")

  nx_graph = nx.Graph()
  nx_graph.add_edges_from(edge.endpoints() for edge in graph.edges.values())

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
  sorted_edges = sorted(solution.edges.values())
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
    description="Run pure SAMR2SSolver on a generated planar graph."
  )
  parser.add_argument("--num-points", type=int, default=20)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--weight", type=int, default=1)
  parser.add_argument("--remove-ratio", type=float, default=0.0)
  parser.add_argument("--use-face-cycle", action="store_true")
  parser.add_argument("--target-k", type=int, default=8)

  # SA Solver Weights
  parser.add_argument("--flow-weight", type=float, default=1.0)
  parser.add_argument("--apsp-weight", type=float, default=1.0)
  parser.add_argument("--disconnected-penalty", type=float, default=10.0)

  # SA Solver Parameters
  parser.add_argument("--initial-temp", type=float, default=5.0)
  parser.add_argument("--final-temp", type=float, default=0.05)
  parser.add_argument("--cooling-rate", type=float, default=0.92)
  parser.add_argument("--sweeps", type=int, default=2)
  parser.add_argument("--restarts", type=int, default=4)
  parser.add_argument("--solver-seed", type=int, default=7)

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
    directed_edges = FaceClusterPartition(target_k=args.target_k).run(graph).get_edges()
    graph.define_edge_direction(set(directed_edges))

  solver = SAMR2SSolver(
    flow_weight=args.flow_weight,
    apsp_weight=args.apsp_weight,
    disconnected_pair_penalty=args.disconnected_penalty,
    initial_temperature=args.initial_temp,
    final_temperature=args.final_temp,
    cooling_rate=args.cooling_rate,
    sweeps_per_temperature=args.sweeps,
    num_restarts=args.restarts,
    random_seed=args.solver_seed,
  )

  solution = solver.run(graph)
  score = Evaluator().run(solution)

  nx_graph = nx.DiGraph()
  nx_graph.add_nodes_from(graph.get_vertices())
  nx_graph.add_edges_from(solution.edges.values())
  print(f"Strongly connected: {nx.is_strongly_connected(nx_graph)}")
  print(f"Original edge count: {original_edge_count}")
  print(f"Final edge count: {len(graph.edges)}")
  print()
  describe_result(graph, score, solution)


if __name__ == "__main__":
  main()
