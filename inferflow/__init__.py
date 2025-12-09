from __future__ import annotations

import importlib
import os
import pathlib
import sys
import typing as t

from inferflow.types import Box as Box
from inferflow.types import ClassificationOutput as ClassificationOutput
from inferflow.types import DetectionOutput as DetectionOutput
from inferflow.types import Device as Device
from inferflow.types import DeviceType as DeviceType
from inferflow.types import O as O
from inferflow.types import P as P
from inferflow.types import Precision as Precision
from inferflow.types import R as R
from inferflow.types import SegmentationOutput as SegmentationOutput

if sys.platform == "win32":
    try:
        import torch

        torch_lib_path = pathlib.Path(torch.__file__).parent / "lib"
        os.add_dll_directory(torch_lib_path.__fspath__())
        os.environ["PATH"] = str(torch_lib_path / os.pathsep / os.environ.get("PATH", ""))
    except ImportError:
        torch = None


HAS_CPP_EXTENSIONS = True

try:
    from inferflow import _C
except ImportError as e:
    _C = None
    HAS_CPP_EXTENSIONS = False
    import warnings as _warnings

    _warnings.warn(f"C++ extensions not available: {e}. Falling back to Python implementation.", stacklevel=2)


__version__ = "0.1.1a0"

__all__ = [
    "_C",
    "HAS_CPP_EXTENSIONS",
    "batch",
    "pipeline",
    "runtime",
    "workflow",
    "types",
    "__version__",
    "P",
    "R",
    "O",
    "DeviceType",
    "Device",
    "Box",
    "ClassificationOutput",
    "DetectionOutput",
    "SegmentationOutput",
    "Precision",
]


# https://peps.python.org/pep-0562/
def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
