from itertools import chain

import networkx as nx

from mr2s_module.protocols import Solution, Score


class ApspSumEvaluator:

  def run(self, solution: Solution) -> Score:
    graph = nx.DiGraph()

    graph.add_edges_from(solution)
    vertices = set(chain.from_iterable(solution))

    total_distance = 0

    # networkx.all_pairs_shortest_path_length returns an iterator of (source, {target: length})
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

    if not nx.is_strongly_connected(graph):
      raise AssertionError('solution must be strongly connected')

    # Sum up all the path lengths
    for source in vertices:
      if source not in path_lengths:
        continue
      for target in vertices:
        if source == target:
          continue

        # If a target is not reachable from source, it won't be in the path_lengths dict.
        # This is the desired behavior - we only sum over existing paths.
        distance = path_lengths[source].get(target)
        if distance is not None:
          total_distance += distance

    return total_distance