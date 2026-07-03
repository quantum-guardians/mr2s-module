import networkx as nx

from mr2s_module.domain import Solution
from mr2s_module.protocols import Graph


def _inverse_distance(edge_id: int, weight: int) -> float:
  if weight <= 0:
    raise ValueError(
      f"Edge {edge_id} weight must be positive to invert, got {weight}"
    )
  return 1.0 / weight


def build_directed_distance_graph(solution: Solution) -> nx.DiGraph:
  """방향화 결과를 거리 = 1/weight 인 DiGraph 로 변환."""
  graph = nx.DiGraph()
  for edge_id, (u, v) in solution.edges.items():
    distance = _inverse_distance(edge_id, solution.graph.edges[edge_id].weight)
    # 같은 방향 평행 간선은 가장 빠른(무거운) 것만 최단거리에 유효.
    if graph.has_edge(u, v):
      distance = min(distance, graph[u][v]["distance"])
    graph.add_edge(u, v, distance=distance)
  graph.add_nodes_from(solution.graph.get_vertices())
  return graph


def build_undirected_distance_graph(graph: Graph) -> nx.Graph:
  """방향화 전 원본 그래프를 거리 = 1/weight 인 무방향 nx.Graph 로 변환."""
  nx_graph = nx.Graph()
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    distance = _inverse_distance(edge.id, edge.weight)
    if nx_graph.has_edge(u, v):
      distance = min(distance, nx_graph[u][v]["distance"])
    nx_graph.add_edge(u, v, distance=distance)
  nx_graph.add_nodes_from(graph.get_vertices())
  return nx_graph


class UndirectedApspCache:
  """무방향 APSP(거리 = 1/weight) 결과를 Graph 인스턴스 단위로 캐시.

  같은 그래프의 solution 을 반복 평가할 때 분모(무방향 기준) 재계산을 없앤다.
  캐시 값에 graph 참조를 함께 들고 있어 id 재사용 충돌이 없다.
  """

  def __init__(self):
    self._cache: dict[int, tuple[Graph, dict[int, dict[int, float]]]] = {}

  def get_lengths(self, graph: Graph) -> dict[int, dict[int, float]]:
    key = id(graph)
    cached = self._cache.get(key)
    if cached is not None and cached[0] is graph:
      return cached[1]

    nx_graph = build_undirected_distance_graph(graph)
    lengths = dict(nx.all_pairs_dijkstra_path_length(nx_graph, weight="distance"))
    self._cache[key] = (graph, lengths)
    return lengths
