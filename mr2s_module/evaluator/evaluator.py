import networkx as nx

from mr2s_module.domain import Score, Solution
from mr2s_module.evaluator.apsp_sum_ranker import ApspSumRanker



class Evaluator:

  @staticmethod
  def _build_graph_from_edges(
      directed_edges: set[tuple[int, int]],
      vertices: set[int],
  ) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(directed_edges)
    graph.add_nodes_from(vertices)
    return graph

  @staticmethod
  def _safe_lookup(sample, var_name: str) -> int:
    try:
      return int(sample[var_name])
    except (KeyError, ValueError):
      return 0

  def _sample_to_directed_edges(
      self,
      sample,
      solution: Solution,
  ) -> set[tuple[int, int]]:
    directed_edges = set()

    for edge in solution.graph.edges:
      if edge.directed:
        directed_edges.add(edge.vertices)
        continue

      bit = self._safe_lookup(sample, edge.to_key())
      if bit == 1:
        directed_edges.add((edge.vertices[1], edge.vertices[0]))
      else:
        directed_edges.add((edge.vertices[0], edge.vertices[1]))

    return directed_edges

  def eval_apsp_sum(self, solution: Solution) -> float:
    return ApspSumRanker().run(solution)

  def eval_strong_connect_rate(self, solution: Solution) -> float:
    vertices = solution.graph.get_vertices()

    if not vertices:
      return 0.0

    total_samples = 0
    strongly_connected_samples = 0

    for datum in solution.sample_set.data(["sample", "num_occurrences"]):
      directed_edges = self._sample_to_directed_edges(datum.sample, solution)
      graph = self._build_graph_from_edges(directed_edges, vertices)
      occurrences = int(datum.num_occurrences)

      total_samples += occurrences
      if nx.is_strongly_connected(graph):
        strongly_connected_samples += occurrences


    if total_samples == 0:
      return 0.0

    return strongly_connected_samples / total_samples

  def eval_flow(self, solution: Solution) -> float:
    incoming_weights: dict[int, float] = {}
    outgoing_weights: dict[int, float] = {}
    vertices = solution.graph.get_vertices()
    edge_weights = {
      edge.id: float(edge.weight)
      for edge in solution.graph.edges
    }

    for source, target in solution.edges:
      weight = edge_weights[(min(source, target), max(source, target))]
      outgoing_weights[source] = outgoing_weights.get(source, 0.0) + weight
      incoming_weights[target] = incoming_weights.get(target, 0.0) + weight

    return float(sum(
      (incoming_weights.get(vertex, 0.0) - outgoing_weights.get(vertex, 0.0)) ** 2
      for vertex in vertices
    ))

  @staticmethod
  def eval_sample_score(solution: Solution) -> float:
    if len(solution.sample_set) == 0:
      return float("inf")
    return float(solution.sample_set.record.energy.min())

  def run(self, solution: Solution) -> Score:
    return Score(
      apsp_sum=self.eval_apsp_sum(solution),
      strong_connect_rate=self.eval_strong_connect_rate(solution),
      flow_score=self.eval_flow(solution),
      sample_score=self.eval_sample_score(solution),
    )
