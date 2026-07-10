from __future__ import annotations

import math
import random

import networkx as nx
from dimod import SampleSet

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.evaluator.distance_util import build_undirected_distance_graph
from mr2s_module.protocols import EvaluatorProtocol
from mr2s_module.util import flow_imbalance


class SAMR2SSolver:
  def __init__(
      self,
      evaluator: EvaluatorProtocol = Evaluator(),
      *,
      apsp_weight: float = 1.0,
      flow_weight: float = 1.0,
      disconnected_pair_penalty: float = 10.0,
      initial_temperature: float = 5.0,
      final_temperature: float = 0.05,
      cooling_rate: float = 0.92,
      sweeps_per_temperature: int = 2,
      num_restarts: int = 4,
      random_seed: int | None = None,
      early_stop_patience: int | None = 3,
      min_temperature_steps: int = 5,
      early_stop_acceptance_rate: float = 0.01,
      min_objective_improvement: float = 0.0,
  ) -> None:
    if initial_temperature <= 0.0:
      raise ValueError("initial_temperature must be positive")
    if final_temperature <= 0.0:
      raise ValueError("final_temperature must be positive")
    if final_temperature >= initial_temperature:
      raise ValueError("final_temperature must be smaller than initial_temperature")
    if not 0.0 < cooling_rate < 1.0:
      raise ValueError("cooling_rate must be in (0.0, 1.0)")
    if sweeps_per_temperature < 1:
      raise ValueError("sweeps_per_temperature must be at least 1")
    if num_restarts < 1:
      raise ValueError("num_restarts must be at least 1")
    if apsp_weight < 0.0:
      raise ValueError("apsp_weight must be non-negative")
    if flow_weight < 0.0:
      raise ValueError("flow_weight must be non-negative")
    if disconnected_pair_penalty < 0.0:
      raise ValueError("disconnected_pair_penalty must be non-negative")
    if early_stop_patience is not None and early_stop_patience < 1:
      raise ValueError("early_stop_patience must be at least 1 or None")
    if min_temperature_steps < 1:
      raise ValueError("min_temperature_steps must be at least 1")
    if not 0.0 <= early_stop_acceptance_rate <= 1.0:
      raise ValueError("early_stop_acceptance_rate must be in [0.0, 1.0]")
    if min_objective_improvement < 0.0:
      raise ValueError("min_objective_improvement must be non-negative")

    self.evaluator = evaluator
    self.apsp_weight = apsp_weight
    self.flow_weight = flow_weight
    self.disconnected_pair_penalty = disconnected_pair_penalty
    self.initial_temperature = initial_temperature
    self.final_temperature = final_temperature
    self.cooling_rate = cooling_rate
    self.sweeps_per_temperature = sweeps_per_temperature
    self.num_restarts = num_restarts
    self.random_seed = random_seed
    self.early_stop_patience = early_stop_patience
    self.min_temperature_steps = min_temperature_steps
    self.early_stop_acceptance_rate = early_stop_acceptance_rate
    self.min_objective_improvement = min_objective_improvement

  @staticmethod
  def _build_direction(
      edge: Edge,
      bit: int,
  ) -> tuple[int, int]:
    if bit == 1:
      return (edge.vertices[1], edge.vertices[0])
    return (edge.vertices[0], edge.vertices[1])

  @classmethod
  def _directed_weighted_edges(
      cls,
      graph: Graph,
      variable_edges: list[Edge],
      state_bits: list[int],
  ) -> list[tuple[int, int, float]]:
    # 간선 단위 (source, target, weight) — pair 키잉 없이 평행 간선 독립 합산.
    triples = [
      (edge.vertices[0], edge.vertices[1], float(edge.weight))
      for edge in graph.edges.values()
      if edge.directed
    ]
    triples.extend(
      (*cls._build_direction(edge, bit), float(edge.weight))
      for edge, bit in zip(variable_edges, state_bits)
    )
    return triples

  @staticmethod
  def _build_graph_weight_scale(graph: Graph, treewidth: float) -> float:
    total_weight = sum(float(edge.weight) for edge in graph.edges.values())
    num_edges = max(1.0, float(len(graph.edges)))
    return max(1.0, (total_weight * total_weight) / (num_edges * treewidth))

  @staticmethod
  def _build_pair_scale(vertices: list[int]) -> float:
    return float(max(1, len(vertices) * max(0, len(vertices) - 1)))

  @staticmethod
  def _build_stretch_and_disconnected_pair_count(
      directed_weighted_edges: list[tuple[int, int, float]],
      vertices: list[int],
      undirected_lengths: dict[int, dict[int, float]],
  ) -> tuple[float, int]:
    # ranker 와 동일한 거리 원시량(1/weight)·stretch(방향화/무방향) 로 통합.
    # 도달 불가 쌍은 stretch 대신 unreachable 로 세어 SA 가 inf 없이 최적화.
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(vertices)
    for u, v, weight in directed_weighted_edges:
      distance = 1.0 / weight
      # 같은 방향 평행 간선은 가장 빠른(무거운) 것만 최단거리에 유효.
      if directed_graph.has_edge(u, v):
        distance = min(distance, directed_graph[u][v]["distance"])
      directed_graph.add_edge(u, v, distance=distance)

    total_stretch = 0.0
    unreachable_pairs = 0
    for source in vertices:
      lengths = nx.single_source_dijkstra_path_length(
        directed_graph, source, weight="distance"
      )
      for target in vertices:
        if source == target:
          continue
        directed_distance = lengths.get(target)
        if directed_distance is None:
          unreachable_pairs += 1
        else:
          total_stretch += directed_distance / undirected_lengths[source][target]

    return total_stretch, unreachable_pairs

  def _objective(
      self,
      graph: Graph,
      variable_edges: list[Edge],
      state_bits: list[int],
      vertices: list[int],
      treewidth: float,
      undirected_lengths: dict[int, dict[int, float]],
  ) -> float:
    directed_weighted_edges = self._directed_weighted_edges(
      graph, variable_edges, state_bits
    )
    apsp_sum, unreachable_pairs = self._build_stretch_and_disconnected_pair_count(
      directed_weighted_edges,
      vertices,
      undirected_lengths,
    )
    flow_score = flow_imbalance(directed_weighted_edges)
    pair_scale = self._build_pair_scale(vertices)
    weight_scale = self._build_graph_weight_scale(graph, treewidth)
    return (
      self.apsp_weight * (apsp_sum / pair_scale)
      + self.flow_weight * (flow_score / weight_scale)
      + self.disconnected_pair_penalty * (float(unreachable_pairs) / pair_scale)
    )

  def _greedy_flow_seed_bits(
      self,
      variable_edges: list[Edge],
      graph: Graph,
  ) -> list[int]:
    balance: dict[int, float] = {}

    for edge in graph.edges.values():
      if not edge.directed:
        continue
      source, target = edge.vertices
      weight = float(edge.weight)
      balance[source] = balance.get(source, 0.0) - weight
      balance[target] = balance.get(target, 0.0) + weight

    def direction_penalty(source: int, target: int, weight: float) -> float:
      source_balance = balance.get(source, 0.0)
      target_balance = balance.get(target, 0.0)
      return (
        (source_balance - weight) ** 2
        + (target_balance + weight) ** 2
        - source_balance ** 2
        - target_balance ** 2
      )

    seed_bits: list[int] = []
    for edge in variable_edges:
      source, target = edge.vertices
      weight = edge.weight
      forward_penalty = direction_penalty(source, target, weight)
      reverse_penalty = direction_penalty(target, source, weight)
      bit = 0 if forward_penalty <= reverse_penalty else 1
      seed_bits.append(bit)

      chosen_source, chosen_target = self._build_direction(edge, bit)
      balance[chosen_source] = balance.get(chosen_source, 0.0) - weight
      balance[chosen_target] = balance.get(chosen_target, 0.0) + weight

    return seed_bits

  def _anneal_bits(
      self,
      graph: Graph,
      variable_edges: list[Edge],
      treewidth: float,
      undirected_lengths: dict[int, dict[int, float]],
  ) -> tuple[list[int], float]:
    if not variable_edges:
      return [], self._objective(
        graph, [], [], sorted(graph.get_vertices()), treewidth, undirected_lengths
      )

    rng = random.Random(self.random_seed)
    vertices = sorted(graph.get_vertices())
    steps_per_temperature = self.sweeps_per_temperature * len(variable_edges)

    best_bits: list[int] | None = None
    best_objective = float("inf")
    seed_bits = self._greedy_flow_seed_bits(variable_edges, graph)
    for restart in range(self.num_restarts):
      if restart == 0:
        current_bits = list(seed_bits)
      else:
        current_bits = [rng.randint(0, 1) for _ in variable_edges]

      current_objective = self._objective(
        graph,
        variable_edges,
        current_bits,
        vertices,
        treewidth,
        undirected_lengths,
      )
      if current_objective < best_objective:
        best_objective = current_objective
        best_bits = list(current_bits)

      restart_best_objective = current_objective
      stale_temperature_steps = 0
      temperature = self.initial_temperature
      temperature_step = 0
      while temperature > self.final_temperature:
        previous_restart_best = restart_best_objective
        accepted_moves = 0
        for _ in range(steps_per_temperature):
          bit_index = rng.randrange(len(variable_edges))
          current_bits[bit_index] ^= 1
          next_objective = self._objective(
            graph,
            variable_edges,
            current_bits,
            vertices,
            treewidth,
            undirected_lengths,
          )
          delta = next_objective - current_objective
          accept = delta <= 0.0 or rng.random() < math.exp(-delta / temperature)
          if accept:
            accepted_moves += 1
            current_objective = next_objective
            restart_best_objective = min(
              restart_best_objective,
              current_objective,
            )
            if current_objective < best_objective:
              best_objective = current_objective
              best_bits = list(current_bits)
          else:
            current_bits[bit_index] ^= 1

        acceptance_rate = accepted_moves / steps_per_temperature
        objective_improvement = previous_restart_best - restart_best_objective
        can_stop = temperature_step + 1 >= self.min_temperature_steps
        is_stale = (
          objective_improvement <= self.min_objective_improvement
          and acceptance_rate <= self.early_stop_acceptance_rate
        )
        stale_temperature_steps = (
          stale_temperature_steps + 1
          if can_stop and is_stale
          else 0
        )
        if (
          self.early_stop_patience is not None
          and stale_temperature_steps >= self.early_stop_patience
        ):
          break

        temperature *= self.cooling_rate
        temperature_step += 1

    if best_bits is None:
      return list(seed_bits), best_objective
    return best_bits, best_objective

  def run(self, graph: Graph) -> Solution:
    # Calculate treewidth approximation once at solver start
    from networkx.algorithms.approximation import treewidth_min_degree
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.get_vertices())
    nx_graph.add_edges_from(edge.endpoints() for edge in graph.edges.values())
    tw, _ = treewidth_min_degree(nx_graph)
    treewidth = max(1.0, float(tw))

    variable_edges = [
      edge
      for edge in graph.edges.values()
      if not edge.directed
    ]

    # ranker stretch 분모(무방향 1/weight APSP)를 run 당 한 번만 계산.
    undirected_lengths = dict(
      nx.all_pairs_dijkstra_path_length(
        build_undirected_distance_graph(graph), weight="distance"
      )
    )

    best_bits, best_objective = self._anneal_bits(
      graph, variable_edges, treewidth, undirected_lengths
    )
    sample = {
      edge.to_key(): bit
      for edge, bit in zip(variable_edges, best_bits)
    }
    sample_set = SampleSet.from_samples(
      [sample],
      vartype="BINARY",
      energy=[best_objective],
      num_occurrences=[1],
    )

    # edge id → 방향. 비트가 variable_edges 와 같은 순서라 평행 간선도 id 별 독립 복원.
    solution_edges: dict[int, tuple[int, int]] = {
      edge.id: self._build_direction(edge, bit)
      for edge, bit in zip(variable_edges, best_bits)
    }
    for edge in graph.edges.values():
      if edge.directed:
        solution_edges[edge.id] = edge.vertices

    solution = Solution(
      edges=solution_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )
    solution.score = self.evaluator.run(solution)
    return solution
