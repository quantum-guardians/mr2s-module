from __future__ import annotations

import networkx as nx
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar, cast

from typing_extensions import TypeIs

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph


EdgeKey = frozenset[int]
EdgeNode = tuple[str, int]
SubdivisionNode = int | EdgeNode
EdgeStep = tuple[int, int, int]
Point = Sequence[float]
PositionMap = Mapping[Hashable, Point]
_EDGE_NODE_KIND = "edge"
_FaceKeyT = TypeVar("_FaceKeyT", bound=Hashable)


def is_edge_node(node: Hashable) -> TypeIs[EdgeNode]:
  """Return True when node is a synthetic domain-edge node."""
  return (
    isinstance(node, tuple)
    and len(node) == 2
    and node[0] == _EDGE_NODE_KIND
    and isinstance(node[1], int)
  )


def domain_graph_to_networkx(graph: Graph) -> nx.Graph:
  """Convert the project Graph model to a simple weighted NetworkX graph.

  Self-loops are ignored because NetworkX planar embedding helpers operate on
  simple planar edges. Duplicate undirected edges are collapsed by keeping the
  smallest weight. This is a planar-only shim: consumers that must see every
  parallel edge should use domain_graph_to_networkx_multi() instead.
  """
  nx_graph = nx.Graph()
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    if u == v:
      continue
    if nx_graph.has_edge(u, v):
      if edge.weight < nx_graph[u][v].get("weight", edge.weight):
        nx_graph[u][v]["weight"] = edge.weight
    else:
      nx_graph.add_edge(u, v, weight=edge.weight)
  return nx_graph


def domain_graph_to_networkx_multi(graph: Graph) -> nx.MultiGraph:
  """Convert the project Graph model to a weighted NetworkX multigraph.

  Every parallel edge is preserved as its own nx edge, keyed by the domain
  edge id so downstream consumers can map results back per edge identity.
  Isolated vertices are kept. Self-loops are skipped because edge
  orientation is meaningless for them.
  """
  nx_graph = nx.MultiGraph()
  nx_graph.add_nodes_from(graph.get_vertices())
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    if u == v:
      continue
    nx_graph.add_edge(u, v, key=edge.id, weight=edge.weight)
  return nx_graph


def domain_graph_to_edge_subdivision(graph: Graph) -> nx.Graph:
  """Convert each domain edge into a synthetic node for planar face traversal.

  A domain edge `(u, v, id=eid)` becomes `u -- ("edge", eid) -- v`.
  Parallel edges therefore remain distinct in simple NetworkX topology while
  still giving planar embedding code a graph it can traverse.
  """
  nx_graph = nx.Graph()
  nx_graph.add_nodes_from(graph.get_vertices())
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    if u == v:
      continue
    edge_node = (_EDGE_NODE_KIND, edge.id)
    nx_graph.add_node(
      edge_node,
      kind=_EDGE_NODE_KIND,
      edge_id=edge.id,
      endpoints=(u, v),
      weight=edge.weight,
    )
    nx_graph.add_edge(u, edge_node, edge_id=edge.id)
    nx_graph.add_edge(edge_node, v, edge_id=edge.id)
  return nx_graph


def normalize_planar_input(
    graph: nx.Graph | nx.PlanarEmbedding,
) -> tuple[nx.Graph, nx.PlanarEmbedding]:
  """Return both a plain graph and a checked planar embedding.

  Use this at public NetworkX-facing boundaries where callers may pass either
  an already-built PlanarEmbedding or a normal graph. Non-planar graphs raise
  ValueError so algorithms do not silently continue with invalid topology.
  """
  if isinstance(graph, nx.PlanarEmbedding):
    embedding = graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(embedding.nodes)
    for u in embedding.nodes:
      for v in embedding.neighbors_cw_order(u):
        if u != v:
          nx_graph.add_edge(u, v)
    embedding.check_structure()
    return nx_graph, embedding

  nx_graph = nx.Graph(graph)
  is_planar, embedding = nx.check_planarity(nx_graph)
  if not is_planar:
    raise ValueError("graph must be planar")
  return nx_graph, cast(nx.PlanarEmbedding, embedding)


def check_planar_embedding(graph: nx.Graph) -> tuple[bool, nx.PlanarEmbedding | None]:
  """Check planarity and return None for the embedding on failure."""
  is_planar, embedding = nx.check_planarity(graph)
  if not is_planar:
    return is_planar, None
  return is_planar, cast(nx.PlanarEmbedding, embedding)


def enumerate_faces(
    graph_or_embedding: nx.Graph | nx.PlanarEmbedding,
) -> list[list[SubdivisionNode]]:
  """Enumerate every face as a cyclic vertex list.

  The outer face is included. Call select_outer_face() or inner_faces() when an
  algorithm needs only bounded/internal faces.
  """
  if isinstance(graph_or_embedding, nx.PlanarEmbedding):
    embedding = graph_or_embedding
  else:
    is_planar, checked_embedding = nx.check_planarity(graph_or_embedding)
    if not is_planar:
      return []
    embedding = cast(nx.PlanarEmbedding, checked_embedding)

  faces: list[list[SubdivisionNode]] = []
  visited: set[tuple[SubdivisionNode, SubdivisionNode]] = set()
  for u in embedding.nodes:
    for v in embedding.neighbors_cw_order(u):
      if (u, v) in visited:
        continue
      faces.append(list(embedding.traverse_face(u, v, mark_half_edges=visited)))
  return faces


def face_edge_steps(face: list[SubdivisionNode] | tuple[SubdivisionNode, ...]) -> list[EdgeStep]:
  """Return `(edge_id, tail, head)` steps from one subdivision face."""
  steps: list[EdgeStep] = []
  for index, node in enumerate(face):
    if not is_edge_node(node):
      continue
    previous_node = face[(index - 1) % len(face)]
    next_node = face[(index + 1) % len(face)]
    if is_edge_node(previous_node) or is_edge_node(next_node):
      raise ValueError(
        "subdivision face must traverse each edge node between two vertices"
      )
    steps.append((node[1], previous_node, next_node))
  return steps


def face_edges(face: list[int] | tuple[int, ...]) -> set[EdgeKey]:
  """Return canonical undirected edge keys for one cyclic face boundary."""
  return {
    frozenset({face[index], face[(index + 1) % len(face)]})
    for index in range(len(face))
  }


def planar_position_map(graph: nx.Graph) -> PositionMap:
  """planar_layout 좌표. 스텁은 값을 ndarray 로 주지만 소비자는 (x, y) 언패킹만 한다."""
  return cast(PositionMap, nx.planar_layout(graph))


def polygon_area(
    face: Sequence[Hashable],
    pos: PositionMap,
) -> float:
  """Return signed polygon area for a face under the supplied 2D positions."""
  area = 0.0
  for index, vertex in enumerate(face):
    next_vertex = face[(index + 1) % len(face)]
    x1, y1 = pos[vertex]
    x2, y2 = pos[next_vertex]
    area += x1 * y2 - x2 * y1
  return area / 2.0


def select_outer_face(faces: list[list[SubdivisionNode]], graph: nx.Graph) -> int:
  """Select the outer face index using planar_layout area as a stable fallback.

  The largest absolute polygon area is treated as the unbounded face. If layout
  generation fails, the face with the most unique vertices is used instead.
  """
  try:
    pos = planar_position_map(graph)
  except nx.NetworkXException:
    return max(range(len(faces)), key=lambda index: len(set(faces[index])))
  return max(
    range(len(faces)),
    key=lambda index: abs(polygon_area(faces[index], pos)),
  )


def inner_faces(
    graph: nx.Graph | Graph,
    pos: PositionMap | None = None,
) -> list[list[SubdivisionNode]]:
  """Enumerate all bounded faces for a NetworkX graph or project Graph.

  Pass original drawing positions when visualization needs the outer face to be
  selected in the same coordinate system as the rendered graph.
  """
  nx_graph = domain_graph_to_networkx(graph) if isinstance(graph, Graph) else nx.Graph(graph)
  faces = enumerate_faces(nx_graph)
  if len(faces) <= 1:
    return []

  if pos is None:
    outer_index = select_outer_face(faces, nx_graph)
  else:
    outer_index = max(
      range(len(faces)),
      key=lambda index: abs(polygon_area(faces[index], pos)),
    )
  return [
    face
    for index, face in enumerate(faces)
    if index != outer_index
  ]


def build_face_edges_map(
    faces: list[list[int]],
) -> dict[EdgeKey, list[int]]:
  """Map each canonical edge key to the face indices that contain it."""
  face_edges_map: dict[EdgeKey, list[int]] = {}
  for face_index, face in enumerate(faces):
    for edge in face_edges(face):
      face_edges_map.setdefault(edge, []).append(face_index)
  return face_edges_map


def build_edge_id_face_edges_map(
    faces: list[list[EdgeStep]],
) -> dict[int, list[int]]:
  """Map each domain edge id to the face indices that contain it."""
  face_edges_map: dict[int, list[int]] = {}
  for face_index, face in enumerate(faces):
    for edge_id in {step[0] for step in face}:
      face_edges_map.setdefault(edge_id, []).append(face_index)
  return face_edges_map


def build_dual_base(
    face_edges_map: Mapping[_FaceKeyT, list[int]],
) -> nx.Graph:
  """Build the face adjacency graph from a face-edge incidence map."""
  dual = nx.Graph()
  for face_indices in face_edges_map.values():
    if len(face_indices) == 2:
      dual.add_edge(face_indices[0], face_indices[1])
  return dual


def clone_edge(edge: Edge) -> Edge:
  """Create a detached Edge with the same orientation and weight."""
  return Edge(edge.vertices[0], edge.vertices[1], edge.weight, edge.directed)


def networkx_to_domain_graph(graph: nx.Graph, *, weight: int = 1) -> Graph:
  """Convert a NetworkX graph to the project Graph model."""
  edge_data = cast(
    Iterable[tuple[int, int, Mapping[str, float]]],
    graph.edges(data=True),
  )
  return Graph(edges=[
    Edge(u, v, int(data.get("weight", weight)), False)
    for u, v, data in edge_data
  ])
