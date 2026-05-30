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
