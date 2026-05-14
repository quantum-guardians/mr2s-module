import os
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from diskcache import Cache

P = ParamSpec("P")
R = TypeVar("R")

_DEFAULT_CACHE_ROOT = os.path.join(
  os.path.expanduser("~"),
  ".cache",
  "mr2s_module",
)


def file_cache(
    *,
    cache_namespace: str,
    cache_key_builder: Callable[P, object],
    cache_directory_resolver: Callable[P, str] | None = None,
    should_cache: Callable[[R], bool] = lambda result: True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
  def decorator(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
      cache_directory = (
        cache_directory_resolver(*args, **kwargs)
        if cache_directory_resolver is not None
        else os.path.join(_DEFAULT_CACHE_ROOT, cache_namespace)
      )
      cache_key = (
        cache_namespace,
        cache_key_builder(*args, **kwargs),
      )

      with Cache(cache_directory) as cache:
        if cache_key in cache:
          return cache[cache_key]

        result = func(*args, **kwargs)
        if should_cache(result):
          cache[cache_key] = result
        return result

    return wrapper

  return decorator
