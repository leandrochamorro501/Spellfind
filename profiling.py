"""Worker measurements isolated by execution context, including concurrent requests."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_current = ContextVar("spellfind_worker_profile", default=None)


@contextmanager
def worker_profile():
    values = {}
    token = _current.set(values)
    try:
        yield values
    finally:
        _current.reset(token)


def add(name, value=1):
    values = _current.get()
    if values is not None:
        values[name] = values.get(name, 0) + value


@contextmanager
def phase(name):
    start = perf_counter()
    try:
        yield
    finally:
        add(name, (perf_counter() - start) * 1000)
