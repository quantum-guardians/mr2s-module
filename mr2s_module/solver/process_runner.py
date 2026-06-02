from dataclasses import dataclass
import multiprocessing
import os
import queue
import signal
import sys
from typing import Callable, Iterable, Literal, TypeVar, cast


ProcessStartMethod = Literal["spawn", "fork"]

_T = TypeVar("_T")
_R = TypeVar("_R")


class ProcessExecutionError(RuntimeError):
  pass


def validate_process_start_method(start_method: str) -> None:
  if start_method not in {"spawn", "fork"}:
    raise ValueError("start_method must be 'spawn' or 'fork'")
  if start_method not in multiprocessing.get_all_start_methods():
    raise ValueError(
      f"start_method {start_method!r} is not supported on this platform"
    )


def default_process_start_method() -> ProcessStartMethod:
  available_methods = multiprocessing.get_all_start_methods()
  if os.name == "nt" or sys.platform == "darwin" or "fork" not in available_methods:
    return "spawn"
  return "fork"


def _prepare_child_process_group() -> None:
  if os.name == "posix" and hasattr(os, "setsid"):
    try:
      os.setsid()
    except OSError:
      pass


def _terminate_process_tree(process: multiprocessing.Process) -> None:
  if not process.is_alive():
    return

  if os.name == "posix":
    try:
      os.killpg(process.pid, signal.SIGTERM)
      return
    except (AttributeError, ProcessLookupError, PermissionError):
      pass
    except OSError:
      pass

  process.terminate()


def _run_process_task(func, index: int, item, result_queue) -> None:
  _prepare_child_process_group()
  try:
    result_queue.put((index, True, func(item)))
  except BaseException as exc:
    result_queue.put((index, False, repr(exc)))


@dataclass(frozen=True)
class ProcessRunner:
  max_workers: int
  start_method: ProcessStartMethod | None = None

  def __post_init__(self) -> None:
    if self.max_workers < 1:
      raise ValueError("max_workers must be at least 1")
    if self.start_method is not None:
      validate_process_start_method(self.start_method)

  def _resolved_start_method(self) -> ProcessStartMethod:
    if self.start_method is not None:
      return self.start_method
    return default_process_start_method()

  def map(self, func: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
    item_list = list(items)
    if not item_list:
      return []

    context = multiprocessing.get_context(self._resolved_start_method())
    result_queue = context.Queue()
    results: list[_R | None] = [None] * len(item_list)
    pending = iter(enumerate(item_list))
    active: dict[int, multiprocessing.Process] = {}
    completed = 0

    def start_available_processes() -> None:
      while len(active) < self.max_workers:
        try:
          index, item = next(pending)
        except StopIteration:
          return

        process = context.Process(
          target=_run_process_task,
          args=(func, index, item, result_queue),
        )
        process.daemon = False
        process.start()
        active[index] = process

    try:
      start_available_processes()
      while completed < len(item_list):
        try:
          index, succeeded, payload = result_queue.get(timeout=0.1)
        except queue.Empty:
          for index, process in list(active.items()):
            if process.exitcode not in (None, 0):
              active.pop(index)
              process.join()
              raise ProcessExecutionError(
                f"process {process.pid} exited with code {process.exitcode}"
              )
          continue

        process = active.pop(index)
        process.join()
        completed += 1
        if not succeeded:
          raise ProcessExecutionError(payload)
        results[index] = payload
        start_available_processes()
    finally:
      for process in active.values():
        _terminate_process_tree(process)
        process.join()
      result_queue.close()

    return [cast(_R, result) for result in results]
