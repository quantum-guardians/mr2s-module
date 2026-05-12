from __future__ import annotations

import math
import random

import networkx as nx
from dimod import SampleSet

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol, FaceCycleProtocol


class SAMR2SSolver:
  def __init__(
      self,
      face_cycle: FaceCycleProtocol | None = None,
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

    self.face_cycle = face_cycle
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

  @staticmethod
  def _build_direction(
      edge: Edge,
      bit: int,
  ) -> tuple[int, int]:
    if bit == 1:
      return (edge.vertices[1], edge.vertices[0])
    return (edge.vertices[0], edge.vertices[1])

  def _state_to_edges(
      self,
      variable_edges: list[Edge],
      state_bits: list[int],
      fixed_edges: set[tuple[int, int]],
  ) -> set[tuple[int, int]]:
    directed_edges = set(fixed_edges)
    directed_edges.update(
      self._build_direction(edge, bit)
      for edge, bit in zip(variable_edges, state_bits)
    )
    return directed_edges

  @staticmethod
  def _build_flow_score(
      directed_edges: set[tuple[int, int]],
      graph: Graph,
  ) -> float:
    incoming_weights: dict[int, float] = {}
    outgoing_weights: dict[int, float] = {}
    edge_weights = {
      edge.id: float(edge.weight)
      for edge in graph.edges
    }

    for source, target in directed_edges:
      weight = edge_weights[(min(source, target), max(source, target))]
      outgoing_weights[source] = outgoing_weights.get(source, 0.0) + weight
      incoming_weights[target] = incoming_weights.get(target, 0.0) + weight

    return float(sum(
      (incoming_weights.get(vertex, 0.0) - outgoing_weights.get(vertex, 0.0)) ** 2
      for vertex in graph.get_vertices()
    ))

  @staticmethod
  def _build_apsp_and_disconnected_pair_count(
      directed_edges: set[tuple[int, int]],
      vertices: list[int],
  ) -> tuple[float, int]:
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(vertices)
    directed_graph.add_edges_from(directed_edges)

    total_distance = 0.0
    unreachable_pairs = 0
    for source in vertices:
      distances = nx.single_source_shortest_path_length(directed_graph, source)
      for target in vertices:
        if source == target:
          continue
        distance = distances.get(target)
        if distance is None:
          unreachable_pairs += 1
        else:
          total_distance += float(distance)

    return total_distance, unreachable_pairs

  def _objective(
      self,
      graph: Graph,
      variable_edges: list[Edge],
      state_bits: list[int],
      fixed_edges: set[tuple[int, int]],
      vertices: list[int],
  ) -> float:
    directed_edges = self._state_to_edges(variable_edges, state_bits, fixed_edges)
    apsp_sum, unreachable_pairs = self._build_apsp_and_disconnected_pair_count(
      directed_edges,
      vertices,
    )
    flow_score = self._build_flow_score(directed_edges, graph)
    return (
      self.apsp_weight * apsp_sum
      + self.flow_weight * flow_score
      + self.disconnected_pair_penalty * float(unreachable_pairs)
    )

  def _greedy_flow_seed_bits(
      self,
      variable_edges: list[Edge],
      fixed_edges: set[tuple[int, int]],
      graph: Graph,
  ) -> list[int]:
    balance: dict[int, float] = {}
    edge_weights = {
      edge.id: float(edge.weight)
      for edge in graph.edges
    }

    for source, target in fixed_edges:
      weight = edge_weights[(min(source, target), max(source, target))]
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
      weight = edge_weights[edge.id]
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
      fixed_edges: set[tuple[int, int]],
  ) -> tuple[list[int], float]:
    if not variable_edges:
      return [], self._objective(graph, [], [], fixed_edges, sorted(graph.get_vertices()))

    rng = random.Random(self.random_seed)
    vertices = sorted(graph.get_vertices())
    steps_per_temperature = self.sweeps_per_temperature * len(variable_edges)

    best_bits: list[int] | None = None
    best_objective = float("inf")
    seed_bits = self._greedy_flow_seed_bits(variable_edges, fixed_edges, graph)

    for restart in range(self.num_restarts):
      if restart == 0:
        current_bits = list(seed_bits)
      else:
        current_bits = [rng.randint(0, 1) for _ in variable_edges]

      current_objective = self._objective(
        graph,
        variable_edges,
        current_bits,
        fixed_edges,
        vertices,
      )
      if current_objective < best_objective:
        best_objective = current_objective
        best_bits = list(current_bits)

      temperature = self.initial_temperature
      while temperature > self.final_temperature:
        for _ in range(steps_per_temperature):
          bit_index = rng.randrange(len(variable_edges))
          current_bits[bit_index] ^= 1
          next_objective = self._objective(
            graph,
            variable_edges,
            current_bits,
            fixed_edges,
            vertices,
          )
          delta = next_objective - current_objective
          accept = delta <= 0.0 or rng.random() < math.exp(-delta / temperature)
          if accept:
            current_objective = next_objective
            if current_objective < best_objective:
              best_objective = current_objective
              best_bits = list(current_bits)
          else:
            current_bits[bit_index] ^= 1
        temperature *= self.cooling_rate

    if best_bits is None:
      return list(seed_bits), best_objective
    return best_bits, best_objective

  def run(self, graph: Graph) -> Solution:
    if self.face_cycle is not None:
      partition = self.face_cycle.run(graph)
      graph.define_edge_direction(set(partition.directed_edges()))

    fixed_edges = {
      edge.vertices
      for edge in graph.edges
      if edge.directed
    }
    variable_edges = [
      edge
      for edge in graph.edges
      if not edge.directed
    ]

    best_bits, best_objective = self._anneal_bits(graph, variable_edges, fixed_edges)
    directed_edges = self._state_to_edges(variable_edges, best_bits, fixed_edges)
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

    solution = Solution(
      edges=directed_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )
    solution.score = self.evaluator.run(solution)
    return solution
