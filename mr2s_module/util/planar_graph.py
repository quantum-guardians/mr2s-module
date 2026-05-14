from __future__ import annotations

from typing import Hashable

import networkx as nx

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph


EdgeKey = tuple[int, int]


def domain_graph_to_networkx(graph: Graph) -> nx.Graph:
  """Convert the project Graph model to a simple weighted NetworkX graph.

  Self-loops are ignored because NetworkX planar embedding helpers operate on
  simple planar edges. Duplicate undirected edges are collapsed by keeping the
  smallest weight.
  """
  nx_graph = nx.Graph()
  for edge in graph.edges:
    u, v = edge.id
    if u == v:
      continue
    if nx_graph.has_edge(u, v):
      if edge.weight < nx_graph[u][v].get("weight", edge.weight):
        nx_graph[u][v]["weight"] = edge.weight
    else:
      nx_graph.add_edge(u, v, weight=edge.weight)
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
  return nx_graph, embedding


def check_planar_embedding(graph: nx.Graph) -> tuple[bool, nx.PlanarEmbedding | None]:
  """Check planarity and return None for the embedding on failure."""
  is_planar, embedding = nx.check_planarity(graph)
  return is_planar, embedding if is_planar else None


def enumerate_faces(graph_or_embedding: nx.Graph | nx.PlanarEmbedding) -> list[list[int]]:
  """Enumerate every face as a cyclic vertex list.

  The outer face is included. Call select_outer_face() or inner_faces() when an
  algorithm needs only bounded/internal faces.
  """
  if isinstance(graph_or_embedding, nx.PlanarEmbedding):
    embedding = graph_or_embedding
  else:
    is_planar, embedding = nx.check_planarity(graph_or_embedding)
    if not is_planar:
      return []

  faces: list[list[int]] = []
  visited: set[tuple[int, int]] = set()
  for u in embedding.nodes:
    for v in embedding.neighbors_cw_order(u):
      if (u, v) in visited:
        continue
      faces.append(list(embedding.traverse_face(u, v, mark_half_edges=visited)))
  return faces


def face_edges(face: list[int] | tuple[int, ...]) -> set[EdgeKey]:
  """Return canonical undirected edge keys for one cyclic face boundary."""
  return {
    tuple(sorted((face[index], face[(index + 1) % len(face)])))
    for index in range(len(face))
  }


def polygon_area(face: list[int], pos: dict[Hashable, object]) -> float:
  """Return signed polygon area for a face under the supplied 2D positions."""
  area = 0.0
  for index, vertex in enumerate(face):
    next_vertex = face[(index + 1) % len(face)]
    x1, y1 = pos[vertex]
    x2, y2 = pos[next_vertex]
    area += x1 * y2 - x2 * y1
  return area / 2.0


def select_outer_face(faces: list[list[int]], graph: nx.Graph) -> int:
  """Select the outer face index using planar_layout area as a stable fallback.

  The largest absolute polygon area is treated as the unbounded face. If layout
  generation fails, the face with the most unique vertices is used instead.
  """
  try:
    pos = nx.planar_layout(graph)
  except nx.NetworkXException:
    return max(range(len(faces)), key=lambda index: len(set(faces[index])))
  return max(
    range(len(faces)),
    key=lambda index: abs(polygon_area(faces[index], pos)),
  )


def inner_faces(
    graph: nx.Graph | Graph,
    pos: dict[int, object] | None = None,
) -> list[list[int]]:
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


def build_dual_base(
    face_edges_map: dict[EdgeKey, list[int]],
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
