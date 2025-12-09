from __future__ import annotations

import importlib
import typing as t

__doctitle__ = "AsyncIO-based InferFlow Modules"

__all__ = ["batch", "pipeline", "runtime", "workflow"]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
