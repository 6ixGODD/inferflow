from __future__ import annotations

try:
    from inferflow import _C

    HAS_CPP_EXTENSIONS = True
except ImportError:
    _C = None
    HAS_CPP_EXTENSIONS = False

__all__ = ["_C", "HAS_CPP_EXTENSIONS"]
__version__ = "0.1.0a2"
