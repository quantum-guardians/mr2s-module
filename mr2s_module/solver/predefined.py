from __future__ import annotations

from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol
from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver


def create_robbin_solver(
    evaluator: EvaluatorProtocol = Evaluator(),
) -> RobbinMR2SSolver:
  """Creates a standalone Robbin MR2S solver."""
  return RobbinMR2SSolver(evaluator=evaluator)


def create_ils_solver(
    max_iter: int = 30,
    patience: int = 5,
    is_relaxed: bool = False,
    perturb_strength: int = 2,
    evaluator: EvaluatorProtocol = Evaluator(),
) -> IlsMR2SSolver:
  """Creates a standalone Iterated Local Search MR2S solver."""
  return IlsMR2SSolver(
    max_iter=max_iter,
    patience=patience,
    is_relaxed=is_relaxed,
    perturb_strength=perturb_strength,
    evaluator=evaluator,
  )
