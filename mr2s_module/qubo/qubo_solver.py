from dimod import SampleSet, SimulatedAnnealingSampler
from dwave.system.composites.embedding import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers.dwave_sampler import DWaveSampler

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol
from mr2s_module.qubo.solution_processing import select_best_sample
from mr2s_module.util.sample_set import empty_binary_sample_set


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
      embedding_child_sampler = DWaveSampler()
      fixed_embedding_child_sampler = DWaveSampler()
      sampler = EmbeddingComposite(embedding_child_sampler)
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
      fixed_embedding_child_sampler=fixed_embedding_child_sampler,
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
      print(
        "SampleSet is empty; "
        f"sample_set.info={sample_set.info}; "
        f"{self._sample_debug_info(sampler, qubo, graph)}"
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

    self.fixed_sampler = FixedEmbeddingComposite(child_sampler, embedding=embedding)
    sample_set = self._sample(self.fixed_sampler, qubo, graph)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
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
