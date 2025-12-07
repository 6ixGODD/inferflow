from __future__ import annotations

import os as _os
import pathlib as _pathlib
import sys as _sys

if _sys.platform == "win32":
    try:
        import torch as _torch

        _torch_lib_path = _pathlib.Path(_torch.__file__).parent / "lib"
        _os.add_dll_directory(_torch_lib_path.__fspath__())
        _os.environ["PATH"] = str(_torch_lib_path / _os.pathsep / _os.environ.get("PATH", ""))
    except ImportError:
        _torch = None

try:
    from inferflow import _C

    HAS_CPP_EXTENSIONS = True
except ImportError as _e:
    _C = None
    HAS_CPP_EXTENSIONS = False
    import warnings as _warnings

    _warnings.warn(f"C++ extensions not available: {_e}. Falling back to Python implementation.", stacklevel=2)

__all__ = ["__init__.pyii", "HAS_CPP_EXTENSIONS"]
__version__ = "0.1.0a2"
