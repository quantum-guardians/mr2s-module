import logging

from dimod import SampleSet, SimulatedAnnealingSampler
from dwave.system.composites.embedding import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers.dwave_sampler import DWaveSampler
import networkx as nx

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol
from mr2s_module.qubo.solution_processing import select_best_sample
from mr2s_module.util.sample_set import empty_binary_sample_set

logger = logging.getLogger(__name__)


class InvalidEmbeddingError(ValueError):
  """Raised when a reused embedding is invalid for the current BQM/target."""


class QuboSolver:
  """QUBO 를 샘플링해 간선 방향(Solution)을 찾는 솔버.

  SA(시뮬레이티드 어닐링)와 QA(D-Wave 양자 어닐러) 백엔드는 사용하는 ``sampler``
  만 다르다. 따라서 ``sampler`` 를 주입받는 단일 클래스로 통합하고, 자주 쓰는 두
  백엔드는 ``create_sa_solver`` / ``create_qa_solver`` 정적 메서드로 생성한다.
  """

  def __init__(
    self,
    ranker: SolutionRankerProtocol,
    sampler,
    num_reads: int | None = None,
    fixed_embedding_child_sampler=None,
  ):
    # num_reads: 샘플러를 몇 번 반복 샘플링할지. None 이면 샘플러 기본값 사용
    self.ranker = ranker
    self.sampler = sampler
    self.num_reads = num_reads
    self.fixed_embedding_child_sampler = fixed_embedding_child_sampler
    self.fixed_sampler = None

  def fixed_embedding_target_graph(self) -> nx.Graph | None:
    child_sampler = self._fixed_embedding_child_sampler()
    if child_sampler is None:
      return None

    to_networkx_graph = getattr(child_sampler, "to_networkx_graph", None)
    if to_networkx_graph is not None:
      return nx.Graph(to_networkx_graph())

    nodelist = getattr(child_sampler, "nodelist", None)
    edgelist = getattr(child_sampler, "edgelist", None)
    if nodelist is None or edgelist is None:
      return None

    target_graph = nx.Graph()
    target_graph.add_nodes_from(nodelist)
    target_graph.add_edges_from(edgelist)
    return target_graph

  @staticmethod
  def create_sa_solver(
    ranker: SolutionRankerProtocol, num_reads: int | None = None
  ) -> "QuboSolver":
    """시뮬레이티드 어닐링 백엔드 솔버 (로컬 실행, 자격증명 불필요)."""
    return QuboSolver(ranker=ranker, sampler=SimulatedAnnealingSampler(), num_reads=num_reads)

  @staticmethod
  def create_qa_solver(
    ranker: SolutionRankerProtocol, num_reads: int = 100
  ) -> "QuboSolver":
    """D-Wave 양자 어닐러 백엔드 솔버 (D-Wave API 자격증명 필요)."""
    try:
      child_sampler = DWaveSampler()
      sampler = EmbeddingComposite(child_sampler)
    except Exception as e:
      raise RuntimeError(
        "D-Wave API credentials not found. "
        "Set DWAVE_API_TOKEN environment variable or configure "
        "~/.config/dwave/dwave.conf"
      ) from e
    return QuboSolver(
      ranker=ranker,
      sampler=sampler,
      num_reads=num_reads,
      fixed_embedding_child_sampler=child_sampler,
    )

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    if not self._has_undirected_edges(graph):
      return self._solution_from_directed_graph(graph)

    sample_set = self._sample(self.sampler, qubo, graph)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )

  def _sample(
      self,
      sampler,
      qubo: QuboMatrix,
      graph: Graph | None = None,
  ) -> SampleSet:
    if self.num_reads is None:
      sample_set = sampler.sample(qubo)
    else:
      sample_set = sampler.sample(qubo, num_reads=self.num_reads)

    if len(sample_set) == 0:
      logger.warning(
        "SampleSet is empty; "
        "sample_set.info=%s; %s",
        sample_set.info,
        self._sample_debug_info(sampler, qubo, graph),
      )

    return sample_set

  def run_with_embedding(
      self,
      qubo: QuboMatrix,
      graph: Graph,
      embedding: dict[object, list[object]],
  ) -> Solution:
    if not self._has_undirected_edges(graph):
      return self._solution_from_directed_graph(graph)

    child_sampler = self.fixed_embedding_child_sampler
    if child_sampler is None:
      child_sampler = getattr(self.sampler, "child", None)
    if child_sampler is None:
      raise NotImplementedError(
        "The configured QUBO sampler does not support fixed embeddings"
      )

    target_graph = self.fixed_embedding_target_graph()
    if target_graph is not None:
      self._validate_embedding(qubo, embedding, target_graph)

    self.fixed_sampler = FixedEmbeddingComposite(child_sampler, embedding=embedding)
    sample_set = self._sample(self.fixed_sampler, qubo, graph)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )

  def _fixed_embedding_child_sampler(self):
    child_sampler = self.fixed_embedding_child_sampler
    if child_sampler is None:
      child_sampler = getattr(self.sampler, "child", None)
    return child_sampler

  def _validate_embedding(
      self,
      qubo: QuboMatrix,
      embedding: dict[object, list[object]],
      target_graph: nx.Graph,
  ) -> None:
    variables = set(getattr(qubo, "variables", []))
    missing_variables = [
      variable
      for variable in variables
      if variable not in embedding or not embedding[variable]
    ]
    if missing_variables:
      raise InvalidEmbeddingError(
        "reused embedding is missing non-empty chains for "
        f"{len(missing_variables)} BQM variables"
      )

    for variable in variables:
      chain = list(embedding[variable])
      missing_nodes = [node for node in chain if node not in target_graph]
      if missing_nodes:
        raise InvalidEmbeddingError(
          f"chain for {variable} contains nodes outside the target graph"
        )
      if len(chain) > 1 and not nx.is_connected(target_graph.subgraph(chain)):
        raise InvalidEmbeddingError(f"chain for {variable} is not connected")

    used_nodes: dict[object, object] = {}
    for variable in variables:
      for node in embedding[variable]:
        previous_variable = used_nodes.get(node)
        if previous_variable is not None:
          raise InvalidEmbeddingError(
            f"chains for {previous_variable} and {variable} overlap at {node}"
          )
        used_nodes[node] = variable

    quadratic = getattr(qubo, "quadratic", {})
    for source, target in quadratic:
      source_chain = embedding.get(source)
      target_chain = embedding.get(target)
      if not source_chain or not target_chain:
        raise InvalidEmbeddingError(
          f"interaction ({source}, {target}) references a missing chain"
        )
      if not any(
          target_graph.has_edge(source_node, target_node)
          for source_node in source_chain
          for target_node in target_chain
      ):
        raise InvalidEmbeddingError(
          f"interaction ({source}, {target}) is not represented on target graph"
        )

  def _has_undirected_edges(self, graph: Graph) -> bool:
    return any(not edge.directed for edge in graph.edges.values())

  def _solution_from_directed_graph(self, graph: Graph) -> Solution:
    return Solution(
      edges={edge.vertices for edge in graph.edges.values()},
      sample_set=empty_binary_sample_set(),
      graph=graph,
      score=None,
    )

  def _sample_debug_info(
      self,
      sampler,
      qubo: QuboMatrix,
      graph: Graph | None,
  ) -> str:
    qubo_variables = getattr(qubo, "variables", None)
    qubo_variable_count = len(qubo_variables) if qubo_variables is not None else "unknown"
    qubo_interactions = getattr(qubo, "quadratic", None)
    qubo_interaction_count = (
      len(qubo_interactions) if qubo_interactions is not None else "unknown"
    )
    debug_parts = [
      f"sampler={sampler.__class__.__name__}",
      f"num_reads={self.num_reads}",
      f"qubo_variables={qubo_variable_count}",
      f"qubo_interactions={qubo_interaction_count}",
    ]
    if graph is not None:
      edge_count = len(graph.edges)
      directed_edge_count = sum(edge.directed for edge in graph.edges.values())
      debug_parts.extend([
        f"graph_vertices={len(graph.get_vertices())}",
        f"graph_edges={edge_count}",
        f"directed_edges={directed_edge_count}",
        f"undirected_edges={edge_count - directed_edge_count}",
      ])
    return "; ".join(debug_parts)
