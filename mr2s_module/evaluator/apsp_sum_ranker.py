import networkx as nx

from mr2s_module.domain import Solution
from mr2s_module.protocols import SolutionRankerProtocol


class ApspSumRanker(SolutionRankerProtocol):

  @staticmethod
  def _build_graph(solution: Solution) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(solution.edges)
    graph.add_nodes_from(solution.graph.get_vertices())
    return graph

  def run(self, solution: Solution) -> float:
    graph = self._build_graph(solution)
    vertices = solution.graph.get_vertices()

    if not nx.is_strongly_connected(graph):
      return float("inf")

    total_distance = 0
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

    for source in vertices:
      if source not in path_lengths:
        continue
      for target in vertices:
        if source == target:
          continue

        distance = path_lengths[source].get(target)
        if distance is not None:
          total_distance += distance

    return float(total_distance)
