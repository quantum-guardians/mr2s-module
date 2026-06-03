import pytest

from mr2s_module import ApspSumRanker
from mr2s_module.qubo import QuboSolver
import mr2s_module.qubo.qubo_solver as qubo_solver_module


def test_missing_credentials_raises_runtime_error(monkeypatch):
  def _raise(*args, **kwargs):
    raise ValueError("no credentials")

  monkeypatch.setattr(qubo_solver_module, "DWaveSampler", _raise)

  with pytest.raises(RuntimeError, match="credentials"):
    QuboSolver.create_qa_solver(ranker=ApspSumRanker())


def test_create_qa_solver_reuses_same_sampler_for_embedding_modes(monkeypatch):
  created_samplers = []

  class FakeDWaveSampler:
    def __init__(self):
      created_samplers.append(self)

  class FakeEmbeddingComposite:
    def __init__(self, child):
      self.child = child

  monkeypatch.setattr(qubo_solver_module, "DWaveSampler", FakeDWaveSampler)
  monkeypatch.setattr(qubo_solver_module, "EmbeddingComposite", FakeEmbeddingComposite)

  solver = QuboSolver.create_qa_solver(ranker=ApspSumRanker())

  assert len(created_samplers) == 1
  assert solver.sampler.child is created_samplers[0]
  assert solver.fixed_embedding_child_sampler is created_samplers[0]
