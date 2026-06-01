from dimod import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

from mr2s_module.domain import Graph, Solution
from mr2s_module.protocols import QuboMatrix, SolutionRankerProtocol
from mr2s_module.qubo.solution_processing import select_best_sample


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
  ):
    # num_reads: 샘플러를 몇 번 반복 샘플링할지. None 이면 샘플러 기본값 사용
    self.ranker = ranker
    self.sampler = sampler
    self.num_reads = num_reads

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
      sampler = EmbeddingComposite(DWaveSampler())
    except Exception as e:
      raise RuntimeError(
        "D-Wave API credentials not found. "
        "Set DWAVE_API_TOKEN environment variable or configure "
        "~/.config/dwave/dwave.conf"
      ) from e
    return QuboSolver(ranker=ranker, sampler=sampler, num_reads=num_reads)

  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution:
    sample_set = self._sample(self.sampler, qubo)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )

  def _sample(self, sampler, qubo: QuboMatrix):
    if self.num_reads is None:
      return sampler.sample(qubo)
    return sampler.sample(qubo, num_reads=self.num_reads)

  def run_with_embedding(
      self,
      qubo: QuboMatrix,
      graph: Graph,
      embedding: dict[object, list[object]],
  ) -> Solution:
    child_sampler = getattr(self.sampler, "child", None)
    if child_sampler is None:
      raise NotImplementedError(
        "The configured QUBO sampler does not support fixed embeddings"
      )

    fixed_sampler = FixedEmbeddingComposite(child_sampler, embedding=embedding)
    sample_set = self._sample(fixed_sampler, qubo)
    return Solution(
      edges=select_best_sample(sample_set, list(graph.edges.values()), self.ranker),
      sample_set=sample_set,
      graph=graph,
      score=None,
    )
