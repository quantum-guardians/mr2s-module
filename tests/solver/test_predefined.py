from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver
from mr2s_module.solver.predefined import (
  create_robbin_solver,
  create_ils_solver,
)


def test_create_robbin_solver() -> None:
  solver = create_robbin_solver()
  assert isinstance(solver, RobbinMR2SSolver)


def test_create_ils_solver() -> None:
  solver = create_ils_solver(max_iter=5)
  assert isinstance(solver, IlsMR2SSolver)
