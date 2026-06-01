import multiprocessing

import pytest

import mr2s_module.solver.process_runner as process_runner
from mr2s_module.solver.process_runner import (
  ProcessRunner,
  default_process_start_method,
  validate_process_start_method,
)


def _increment(value: int) -> int:
  return value + 1


def _put_increment(value: int, result_queue) -> None:
  result_queue.put(value + 1)


def _run_nested_process(value: int) -> int:
  context = multiprocessing.get_context("spawn")
  result_queue = context.Queue()
  process = context.Process(
    target=_put_increment,
    args=(value, result_queue),
  )
  process.start()
  result = result_queue.get(timeout=5)
  process.join()
  if process.exitcode != 0:
    raise RuntimeError(f"nested process failed with code {process.exitcode}")
  return result


def test_validate_process_start_method_rejects_unknown_method() -> None:
  with pytest.raises(ValueError, match="start_method"):
    validate_process_start_method("forkserver")


def test_validate_process_start_method_rejects_unsupported_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    process_runner.multiprocessing,
    "get_all_start_methods",
    lambda: ["spawn"],
  )

  with pytest.raises(ValueError, match="not supported"):
    validate_process_start_method("fork")


def test_default_process_start_method_uses_spawn_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(process_runner.os, "name", "nt")
  monkeypatch.setattr(process_runner.sys, "platform", "win32")
  monkeypatch.setattr(
    process_runner.multiprocessing,
    "get_all_start_methods",
    lambda: ["spawn"],
  )

  assert default_process_start_method() == "spawn"


def test_default_process_start_method_uses_spawn_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(process_runner.os, "name", "posix")
  monkeypatch.setattr(process_runner.sys, "platform", "darwin")
  monkeypatch.setattr(
    process_runner.multiprocessing,
    "get_all_start_methods",
    lambda: ["spawn", "fork"],
  )

  assert default_process_start_method() == "spawn"


def test_default_process_start_method_uses_fork_on_other_posix_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(process_runner.os, "name", "posix")
  monkeypatch.setattr(process_runner.sys, "platform", "linux")
  monkeypatch.setattr(
    process_runner.multiprocessing,
    "get_all_start_methods",
    lambda: ["fork", "spawn"],
  )

  assert default_process_start_method() == "fork"


def test_process_runner_rejects_non_positive_worker_count() -> None:
  with pytest.raises(ValueError, match="max_workers"):
    ProcessRunner(max_workers=0)


def test_process_runner_uses_configured_context() -> None:
  if "fork" not in multiprocessing.get_all_start_methods():
    pytest.skip("fork start method is not supported on this platform")

  result = ProcessRunner(max_workers=2, start_method="fork").map(
    _increment,
    [1, 2],
  )

  assert result == [2, 3]


def test_process_runner_uses_default_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(process_runner, "default_process_start_method", lambda: "spawn")

  result = ProcessRunner(max_workers=2).map(
    _increment,
    [1, 2],
  )

  assert result == [2, 3]


def test_process_runner_allows_nested_multiprocessing() -> None:
  result = ProcessRunner(max_workers=2, start_method="spawn").map(
    _run_nested_process,
    [1, 2],
  )

  assert result == [2, 3]
