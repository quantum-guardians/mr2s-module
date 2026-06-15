from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import Edge, Graph, SAMR2SSolver, SATemperatureTrace
from tests.util.graph_fixtures import delaunay_graph


def thin_biconnected_graph(
    graph: Graph,
    remove_ratio: float,
    seed: int,
) -> Graph:
  if not 0.0 <= remove_ratio < 1.0:
    raise ValueError("remove_ratio must be in [0.0, 1.0)")

  nx_graph = nx.Graph()
  nx_graph.add_nodes_from(graph.get_vertices())
  nx_graph.add_edges_from(edge.endpoints() for edge in graph.edges.values())

  target_edge_count = round(nx_graph.number_of_edges() * (1.0 - remove_ratio))
  candidate_edges = list(nx_graph.edges())
  np.random.default_rng(seed).shuffle(candidate_edges)

  for edge in candidate_edges:
    if nx_graph.number_of_edges() <= target_edge_count:
      break
    nx_graph.remove_edge(*edge)
    if not nx.is_biconnected(nx_graph):
      nx_graph.add_edge(*edge)

  return Graph(edges=[
    Edge(u, v, graph.edges[frozenset({u, v})].weight, False)
    for u, v in nx_graph.edges()
  ])


def write_trace_csv(path: Path, traces: list[SATemperatureTrace]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(asdict(traces[0])))
    writer.writeheader()
    writer.writerows(asdict(trace) for trace in traces)


def plot_traces(path: Path, traces: list[SATemperatureTrace]) -> None:
  iterations = [trace.total_iterations for trace in traces]
  series = [
    ("APSP sum", [trace.apsp_sum for trace in traces]),
    ("Flow score", [trace.flow_score for trace in traces]),
    ("Unreachable ordered pairs", [trace.unreachable_pairs for trace in traces]),
    ("Objective", [trace.objective for trace in traces]),
    ("Acceptance rate", [trace.acceptance_rate for trace in traces]),
  ]

  figure, axes = plt.subplots(len(series), 1, figsize=(12, 15), sharex=True)
  for axis, (title, values) in zip(axes, series):
    axis.plot(iterations, values, linewidth=1.5)
    axis.set_ylabel(title)
    axis.grid(alpha=0.3)

  axes[-1].set_xlabel("Completed edge-flip iterations")
  figure.suptitle("SAMR2SSolver metrics by temperature step")
  figure.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Plot direct-SA metrics for a thinned 200-vertex planar graph."
  )
  parser.add_argument("--vertices", type=int, default=200)
  parser.add_argument("--remove-ratio", type=float, default=0.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-restarts", type=int, default=1)
  parser.add_argument("--sweeps-per-temperature", type=int, default=1)
  parser.add_argument("--cooling-rate", type=float, default=0.8)
  parser.add_argument("--early-stop-patience", type=int, default=3)
  parser.add_argument("--min-temperature-steps", type=int, default=5)
  parser.add_argument("--early-stop-acceptance-rate", type=float, default=0.01)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/sa-metrics"),
  )
  args = parser.parse_args()

  original_graph = delaunay_graph(args.vertices, args.seed)
  graph = thin_biconnected_graph(
    original_graph,
    remove_ratio=args.remove_ratio,
    seed=args.seed,
  )
  traces: list[SATemperatureTrace] = []

  started_at = time.perf_counter()
  solution = SAMR2SSolver(
    random_seed=args.seed,
    num_restarts=args.num_restarts,
    sweeps_per_temperature=args.sweeps_per_temperature,
    cooling_rate=args.cooling_rate,
    trace_callback=traces.append,
    early_stop_patience=args.early_stop_patience,
    min_temperature_steps=args.min_temperature_steps,
    early_stop_acceptance_rate=args.early_stop_acceptance_rate,
  ).run(graph)
  elapsed_seconds = time.perf_counter() - started_at

  if not traces:
    raise RuntimeError("SA produced no temperature traces")

  csv_path = args.output_dir / "sa_metrics.csv"
  plot_path = args.output_dir / "sa_metrics.png"
  write_trace_csv(csv_path, traces)
  plot_traces(plot_path, traces)

  final_graph = nx.DiGraph()
  final_graph.add_nodes_from(graph.get_vertices())
  final_graph.add_edges_from(solution.edges)
  undirected_graph = nx.Graph()
  undirected_graph.add_nodes_from(graph.get_vertices())
  undirected_graph.add_edges_from(edge.endpoints() for edge in graph.edges.values())

  removed_edges = len(original_graph.edges) - len(graph.edges)
  print(f"vertices={len(graph.get_vertices())}")
  print(f"original_edges={len(original_graph.edges)}")
  print(f"remaining_edges={len(graph.edges)}")
  print(f"removed_edges={removed_edges}")
  print(f"actual_remove_ratio={removed_edges / len(original_graph.edges):.4f}")
  print(f"biconnected={nx.is_biconnected(undirected_graph)}")
  print(f"final_strongly_connected={nx.is_strongly_connected(final_graph)}")
  print(f"temperature_steps={len(traces)}")
  print(f"edge_flip_iterations={traces[-1].total_iterations}")
  print(f"stopped_early={traces[-1].stopped_early}")
  print(f"elapsed_seconds={elapsed_seconds:.3f}")
  print(f"csv={csv_path.resolve()}")
  print(f"plot={plot_path.resolve()}")


if __name__ == "__main__":
  main()
