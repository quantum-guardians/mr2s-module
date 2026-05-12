import random
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mr2s_module.solver.dnc_mr2s_solver as dnc_mr2s_solver
from mr2s_module.cycle import BalancedFaceGraphClusterer, FaceCycle
from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import ApspSumRanker, Evaluator
from mr2s_module.qubo import (
  FlowPolyGenerator,
  NHop,
  NHopPolyGenerator,
  SAQuboSolver,
  SmallWorldSpec,
)
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import (
  build_dual_base,
  build_face_edges_map,
  clone_edge,
  domain_graph_to_networkx,
  enumerate_faces,
  polygon_area,
)
from tests.util.graph_fixtures import delaunay_graph_with_pos

_OUTPUT_DIR = Path(__file__).parent / "output"
pytest.importorskip("scipy.spatial")
_PALETTE = plt.colormaps.get_cmap("tab20")


class ConfiguredSAQuboSolver(SAQuboSolver):
  def __init__(self, num_reads: int) -> None:
    super().__init__(ranker=ApspSumRanker())
    self.num_reads = num_reads

  def run(self, qubo, graph: Graph) -> Solution:
    sample_set = self.sampler.sample(qubo, num_reads=self.num_reads)
    return Solution(
      edges=self._select_best_sample(sample_set, graph.edges),
      graph=graph,
      sample_set=sample_set,
      score=None,
    )


def build_grid_planar_graph(rows: int, cols: int, weight: int = 1) -> Graph:
  nx_graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
  return Graph(edges=[
    Edge(min(u, v), max(u, v), weight, False)
    for u, v in nx_graph.edges()
  ])


def build_delaunay_planar_graph(
    num_points: int,
    seed: int,
    weight: int = 1,
) -> tuple[Graph, dict[int, tuple[float, float]]]:
  graph, raw_positions = delaunay_graph_with_pos(
    num_points,
    seed,
    weight=weight,
  )
  positions = {
    vertex: (float(point[0]), float(point[1]))
    for vertex, point in raw_positions.items()
  }
  return graph, positions


def remove_edges_by_percent(
    graph: Graph,
    remove_percent: int,
    seed: int = 7,
    weight: int = 1,
) -> tuple[Graph, int]:
  if not 0 <= remove_percent < 100:
    raise ValueError("remove_percent must be in [0, 100).")

  nx_graph = nx.Graph()
  nx_graph.add_edges_from(edge.id for edge in graph.edges)

  edge_count = nx_graph.number_of_edges()
  target_remove_count = round(edge_count * remove_percent / 100)
  shuffled_edges = list(nx_graph.edges())
  random.Random(seed).shuffle(shuffled_edges)

  removed_count = 0
  for u, v in shuffled_edges:
    if removed_count >= target_remove_count:
      break

    nx_graph.remove_edge(u, v)
    if nx.is_biconnected(nx_graph):
      removed_count += 1
    else:
      nx_graph.add_edge(u, v)

  thinned_graph = Graph(edges=[
    Edge(min(u, v), max(u, v), weight, False)
    for u, v in nx_graph.edges()
  ])
  return thinned_graph, removed_count


def build_dnc_solver(num_reads: int = 20) -> DnCMr2sSolver:
  n_hop_generator = NHopPolyGenerator()
  n_hop_generator.small_world_spec = SmallWorldSpec(
    n_hops=[NHop(n=2, weight=1)]
  )
  mr2s_solver = build_qubo_solver(num_reads)
  return DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    face_cycle=FaceCycle(
      target_k=4,
      clusterer=BalancedFaceGraphClusterer(),
    ),
  )


def build_qubo_solver(num_reads: int = 20) -> QuboMR2SSolver:
  return QuboMR2SSolver(
    qubo_solver=ConfiguredSAQuboSolver(num_reads=num_reads),
  )


def clone_graph(graph: Graph) -> Graph:
  return Graph(edges=[clone_edge(edge) for edge in graph.edges])


def _trace_entry(
    depth: int,
    graph: Graph,
    status: str,
    sub_graphs: list[Graph] | None = None,
) -> dict[str, object]:
  sub_graphs = sub_graphs or []
  return {
    "depth": depth,
    "status": status,
    "vertices": len(graph.get_vertices()),
    "edges": len(graph.edges),
    "subgraph_count": len(sub_graphs),
    "subgraphs": [
      {
        "vertices": len(sub_graph.get_vertices()),
        "edges": len(sub_graph.edges),
        "directed_edges": len([
          edge for edge in sub_graph.edges if edge.directed
        ]),
        "undirected_edges": len([
          edge for edge in sub_graph.edges if not edge.directed
        ]),
      }
      for sub_graph in sub_graphs
    ],
  }


def trace_division(
    solver: DnCMr2sSolver,
    graph: Graph,
    depth: int = 0,
) -> tuple[list[dict[str, object]], list[Graph]]:
  if solver._can_embed(graph):
    return [_trace_entry(depth, graph, "embedded")], [graph]

  sub_graphs = solver.divide_graph(graph)
  if len(sub_graphs) == 1 and sub_graphs[0] is graph:
    return [_trace_entry(depth, graph, "fallback", sub_graphs)], [graph]

  return [_trace_entry(depth, graph, "divided", sub_graphs)], sub_graphs


def print_division_trace(trace: list[dict[str, object]]) -> None:
  print("DnC division trace")
  for entry in trace:
    indent = "  " * int(entry["depth"])
    print(
      f"{indent}- depth={entry['depth']} "
      f"status={entry['status']} "
      f"vertices={entry['vertices']} "
      f"edges={entry['edges']} "
      f"subgraph_count={entry['subgraph_count']}"
    )
    for index, sub_graph in enumerate(entry["subgraphs"]):
      print(
        f"{indent}  subgraph[{index}] "
        f"vertices={sub_graph['vertices']} "
        f"edges={sub_graph['edges']} "
        f"directed_edges={sub_graph['directed_edges']} "
        f"undirected_edges={sub_graph['undirected_edges']}"
      )
  print()


def print_leaf_sub_graph_summary(sub_graphs: list[Graph]) -> None:
  print("DnC leaf subgraphs")
  print(f"  leaf_count: {len(sub_graphs)}")
  for index, sub_graph in enumerate(sub_graphs):
    directed_edges = [edge for edge in sub_graph.edges if edge.directed]
    undirected_edges = [edge for edge in sub_graph.edges if not edge.directed]
    edge_preview = [
      edge.vertices if edge.directed else edge.id
      for edge in sub_graph.edges[:10]
    ]

    print(f"  leaf[{index}]")
    print(f"    vertices: {len(sub_graph.get_vertices())}")
    print(f"    edges: {len(sub_graph.edges)}")
    print(f"    directed_edges: {len(directed_edges)}")
    print(f"    undirected_edges: {len(undirected_edges)}")
    print(f"    edge_preview: {edge_preview}")
  print()

def _draw_edges(ax, graph: Graph, positions, color: str, linewidth: float) -> None:
  for edge in graph.edges:
    u, v = edge.id
    ax.plot(
      [positions[u][0], positions[v][0]],
      [positions[u][1], positions[v][1]],
      color=color,
      linewidth=linewidth,
      alpha=0.65,
      zorder=1,
    )


def _draw_boundary_edges(
    ax,
    boundary_edges: set[tuple[int, int]],
    positions,
    color: str = "#111111",
    linewidth: float = 2.0,
) -> None:
  for u, v in boundary_edges:
    ax.plot(
      [positions[u][0], positions[v][0]],
      [positions[u][1], positions[v][1]],
      color=color,
      linewidth=linewidth,
      alpha=0.95,
      zorder=4,
    )


def _draw_vertices(ax, graph: Graph, positions) -> None:
  for vertex in sorted(graph.get_vertices()):
    ax.scatter(
      positions[vertex][0],
      positions[vertex][1],
      color="#222222",
      s=14,
      zorder=5,
    )


def _save_graph_png(
    graph: Graph,
    positions,
    path: Path,
    title: str,
) -> None:
  fig, ax = plt.subplots(figsize=(10, 8))
  _draw_edges(ax, graph, positions, color="#555555", linewidth=1.1)
  directed_graph = Graph(edges=[edge for edge in graph.edges if edge.directed])
  _draw_edges(ax, directed_graph, positions, color="#d14", linewidth=2.0)
  _draw_vertices(ax, graph, positions)
  ax.set_title(title)
  ax.set_aspect("equal")
  ax.axis("off")
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def _cluster_groups(face_to_cluster: dict[int, int]) -> list[list[int]]:
  groups: dict[int, list[int]] = {}
  for face_idx, cluster_id in face_to_cluster.items():
    groups.setdefault(cluster_id, []).append(face_idx)
  return [groups[cluster_id] for cluster_id in sorted(groups)]


def _save_face_groups_png(
    graph: Graph,
    positions,
    inner_faces: list[list[int]],
    groups: list[list[int]],
    boundary_edges: set[tuple[int, int]],
    path: Path,
    title: str,
) -> None:
  fig, ax = plt.subplots(figsize=(10, 8))
  _draw_edges(ax, graph, positions, color="#aaaaaa", linewidth=0.8)

  for group_id, group in enumerate(groups):
    color = _PALETTE(group_id % 20)
    for face_idx in group:
      polygon = plt.Polygon(
        [positions[vertex] for vertex in inner_faces[face_idx]],
        facecolor=color,
        edgecolor="none",
        alpha=0.50,
        zorder=2,
      )
      ax.add_patch(polygon)

  _draw_boundary_edges(ax, boundary_edges, positions)
  _draw_vertices(ax, graph, positions)
  ax.set_title(title)
  ax.set_aspect("equal")
  ax.axis("off")
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def build_partition_diagnostic(graph: Graph, face_cycle: FaceCycle) -> dict:
  nx_graph = domain_graph_to_networkx(graph)
  component = face_cycle._extract_biconnected_components(nx_graph)[0]
  planar_positions = nx.planar_layout(component)
  raw_faces = enumerate_faces(component)
  outer_idx = int(np.argmax([
    abs(polygon_area(face, planar_positions))
    for face in raw_faces
  ]))
  inner_faces = [face for idx, face in enumerate(raw_faces) if idx != outer_idx]
  face_edges_map = build_face_edges_map(inner_faces)
  centroids = [
    np.mean([planar_positions[vertex] for vertex in face], axis=0)
    for face in inner_faces
  ]
  dual_base = build_dual_base(face_edges_map)

  target_k = max(1, min(face_cycle.target_k, len(inner_faces)))
  face_to_cluster = face_cycle.clusterer.run(centroids, dual_base, target_k)
  boundary_edges, outer_edges = FaceCycle._collect_boundary_edges(
    face_edges_map,
    face_to_cluster,
  )
  repair_edges = FaceCycle._wall_protected_repair(
    component,
    boundary_edges,
    outer_edges,
  )
  final_boundary = face_cycle._apply_boundary_repair(
    boundary_edges,
    repair_edges,
  )

  face_graph = nx.Graph()
  face_graph.add_nodes_from(range(len(inner_faces)))
  for edge, face_indices in face_edges_map.items():
    if len(face_indices) == 2 and edge not in final_boundary:
      face_graph.add_edge(face_indices[0], face_indices[1])
  true_components = FaceCycle._filter_ghost_components(
    face_graph,
    inner_faces,
    outer_edges,
    final_boundary,
  )

  return {
    "inner_faces": inner_faces,
    "pre_groups": _cluster_groups(face_to_cluster),
    "pre_boundary": boundary_edges,
    "post_groups": true_components,
    "post_boundary": final_boundary,
  }


def write_partition_pngs(
    graph: Graph,
    sub_graphs: list[Graph],
    face_cycle: FaceCycle,
    positions,
) -> list[Path]:
  _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  diagnostic = build_partition_diagnostic(graph, face_cycle)
  paths = [
    _OUTPUT_DIR / f"dnc_{face_cycle.repair_mode}_repair_graph_full.png",
    _OUTPUT_DIR / f"dnc_{face_cycle.repair_mode}_repair_pre_clusters.png",
    _OUTPUT_DIR / f"dnc_{face_cycle.repair_mode}_repair_post_components.png",
  ]

  _save_graph_png(
    graph=graph,
    positions=positions,
    path=paths[0],
    title=f"full graph vertices={len(graph.get_vertices())} edges={len(graph.edges)}",
  )
  _save_face_groups_png(
    graph=graph,
    positions=positions,
    inner_faces=diagnostic["inner_faces"],
    groups=diagnostic["pre_groups"],
    boundary_edges=diagnostic["pre_boundary"],
    path=paths[1],
    title=f"pre-repair face clusters ({face_cycle.repair_mode})",
  )
  _save_face_groups_png(
    graph=graph,
    positions=positions,
    inner_faces=diagnostic["inner_faces"],
    groups=diagnostic["post_groups"],
    boundary_edges=diagnostic["post_boundary"],
    path=paths[2],
    title=f"post-repair true components ({face_cycle.repair_mode})",
  )

  for index, sub_graph in enumerate(sub_graphs):
    path = _OUTPUT_DIR / f"dnc_{face_cycle.repair_mode}_repair_leaf_subgraph_{index}.png"
    _save_graph_png(
      graph=sub_graph,
      positions=positions,
      path=path,
      title=(
        f"leaf[{index}] vertices={len(sub_graph.get_vertices())} "
        f"edges={len(sub_graph.edges)}"
      ),
    )
    paths.append(path)

  return paths


def run_timed_solver(name: str, solver, graph: Graph) -> dict[str, object]:
  started_at = time.perf_counter()
  solution = solver.run(graph)
  elapsed_seconds = time.perf_counter() - started_at
  score = solution.score
  selected_undirected_edges = {
    (min(source, target), max(source, target))
    for source, target in solution.edges
  }
  input_edge_ids = {edge.id for edge in graph.edges}

  return {
    "name": name,
    "solution": solution,
    "elapsed_seconds": elapsed_seconds,
    "vertices": len(graph.get_vertices()),
    "edges": len(graph.edges),
    "selected_edges": len(solution.edges),
    "selected_undirected_edges": len(selected_undirected_edges),
    "reverse_direction_conflicts": (
      len(solution.edges) - len(selected_undirected_edges)
    ),
    "missing_input_edges": len(input_edge_ids - selected_undirected_edges),
    "extra_selected_edges": len(selected_undirected_edges - input_edge_ids),
    "apsp_sum": score.apsp_sum if score is not None else None,
    "strong_connect_rate": (
      score.strong_connect_rate if score is not None else None
    ),
    "flow_score": score.flow_score if score is not None else None,
    "sample_score": score.sample_score if score is not None else None,
  }


def print_solver_comparison(results: list[dict[str, object]]) -> None:
  print()
  print("MR2S solver comparison")
  for result in results:
    print(f"  {result['name']}")
    print(f"    elapsed_seconds: {result['elapsed_seconds']:.4f}")
    print(f"    vertices: {result['vertices']}")
    print(f"    edges: {result['edges']}")
    print(f"    selected_edges: {result['selected_edges']}")
    print(f"    selected_undirected_edges: {result['selected_undirected_edges']}")
    print(f"    reverse_direction_conflicts: {result['reverse_direction_conflicts']}")
    print(f"    missing_input_edges: {result['missing_input_edges']}")
    print(f"    extra_selected_edges: {result['extra_selected_edges']}")
    print(f"    apsp_sum: {result['apsp_sum']}")
    print(f"    strong_connect_rate: {result['strong_connect_rate']}")
    print(f"    flow_score: {result['flow_score']}")
    print(f"    sample_score: {result['sample_score']}")


@pytest.mark.slow
def test_compare_dnc_mr2s_solver_and_qubo_mr2s_solver_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  base_graph, _positions = build_delaunay_planar_graph(
    num_points=1000,
    seed=11,
  )
  graph, removed_count = remove_edges_by_percent(
    graph=base_graph,
    remove_percent=50,
    seed=11,
  )

  def estimate_with_test_limit(bqm):
    if len(bqm.variables) > 35:
      raise RuntimeError("test limit exceeded")
    return None

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_with_test_limit,
  )

  dnc_solver = build_dnc_solver(num_reads=5)
  qubo_solver = build_qubo_solver(num_reads=5)
  division_trace, leaf_sub_graphs = trace_division(dnc_solver, clone_graph(graph))
  results = [
    run_timed_solver("dnc_mr2s_solver", dnc_solver, clone_graph(graph)),
    run_timed_solver("qubo_mr2s_solver", qubo_solver, clone_graph(graph)),
  ]

  print_solver_comparison(results)
  print("  dnc_division")
  print(f"    original_edges: {len(base_graph.edges)}")
  print(f"    removed_edges: {removed_count}")
  print(f"    leaf_count: {len(leaf_sub_graphs)}")
  print(f"    trace_entries: {len(division_trace)}")

  assert len(graph.edges) == len(base_graph.edges) - removed_count
  assert len(leaf_sub_graphs) >= 1
  assert all(result["elapsed_seconds"] >= 0.0 for result in results)
  assert all(result["selected_edges"] > 0 for result in results)
  assert all(result["reverse_direction_conflicts"] == 0 for result in results)
  assert all(result["missing_input_edges"] == 0 for result in results)
  assert all(result["extra_selected_edges"] == 0 for result in results)
  assert all(result["apsp_sum"] is not None for result in results)
  assert all(result["strong_connect_rate"] is not None for result in results)
  assert all(result["flow_score"] is not None for result in results)
  assert all(result["sample_score"] is not None for result in results)


@pytest.mark.slow
def test_run_dnc_mr2s_solver_on_planar_graph_with_removed_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  remove_percent = 20
  base_graph, positions = build_delaunay_planar_graph(
    num_points=1000,
    seed=7,
  )
  graph, removed_count = remove_edges_by_percent(
    graph=base_graph,
    remove_percent=remove_percent,
  )

  def estimate_with_test_limit(bqm):
    if len(bqm.variables) > 100:
      raise RuntimeError("test limit exceeded")
    return None

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_with_test_limit,
  )

  solver = build_dnc_solver()
  division_trace, leaf_sub_graphs = trace_division(solver, graph)
  solution = solver.run(graph)
  print_division_trace(division_trace)
  print_leaf_sub_graph_summary(leaf_sub_graphs)
  png_paths = write_partition_pngs(
    graph=graph,
    sub_graphs=leaf_sub_graphs,
    face_cycle=solver.face_cycle,
    positions=positions,
  )
  print("DnC partition PNGs")
  for path in png_paths:
    print(f"  {path}")
  print()

  directed_graph = nx.DiGraph()
  directed_graph.add_nodes_from(graph.get_vertices())
  directed_graph.add_edges_from(solution.edges)

  print()
  print("DnC MR2S planar graph performance")
  print(f"  remove_percent: {remove_percent}")
  print(f"  original_edges: {len(base_graph.edges)}")
  print(f"  removed_edges: {removed_count}")
  print(f"  final_edges: {len(graph.edges)}")
  print(f"  vertices: {len(graph.get_vertices())}")
  print(f"  selected_edges: {len(solution.edges)}")
  print(f"  strongly_connected: {nx.is_strongly_connected(directed_graph)}")
  if solution.score is not None:
    print(f"  apsp_sum: {solution.score.apsp_sum}")
    print(f"  strong_connect_rate: {solution.score.strong_connect_rate:.4f}")
    print(f"  flow_score: {solution.score.flow_score}")
    print(f"  sample_score: {solution.score.sample_score}")
  print(f"  directed_edges_preview: {sorted(solution.edges)[:20]}")

  assert len(graph.edges) == len(base_graph.edges) - removed_count
  assert solution.graph is graph
  assert len(solution.edges) > 0
  assert len(leaf_sub_graphs) >= 1
  assert all(path.exists() and path.stat().st_size > 0 for path in png_paths)
