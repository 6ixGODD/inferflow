#!/usr/bin/env python
"""Setup script for InferFlow.

C++ extensions are optional and only built when:
1. Explicitly enabled via INFERFLOW_BUILD_CPP=1
2. In development mode (pip install -e .)
3. In CI/CD environments

For regular installation:  pip install inferflow
For development:  INFERFLOW_BUILD_CPP=1 pip install -e .
"""

import os
import pathlib
import sys

from setuptools import setup

HERE = pathlib.Path(__file__).parent.resolve()


def should_build_cpp_extension() -> bool:
    """Determine if C++ extension should be built.

    Returns True if:
    - INFERFLOW_BUILD_CPP=1 is set
    - Installing in development mode (-e/--editable)
    - Running from source tree (not sdist/wheel)
    """
    # Explicit environment variable
    if os.environ.get("INFERFLOW_BUILD_CPP", "0") == "1":
        print("🔧 C++ extension build:  ENABLED (via INFERFLOW_BUILD_CPP=1)")
        return True

    # Explicit disable
    if os.environ.get("INFERFLOW_BUILD_CPP", "0") == "0":
        print("⏭️  C++ extension build:  DISABLED (via INFERFLOW_BUILD_CPP=0)")
        return False

    # Check if in development mode
    if "develop" in sys.argv or any(arg.startswith("--editable") for arg in sys.argv):
        print("🔧 C++ extension build:  ENABLED (development mode)")
        return True

    # Check if in CI/CD
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        print("🔧 C++ extension build: ENABLED (CI/CD mode)")
        return True

    # Default:  disabled for regular pip install
    print("⏭️  C++ extension build: DISABLED (regular install)")
    print("   💡 To enable:  INFERFLOW_BUILD_CPP=1 pip install -e .")
    return False


def get_cpp_extension():
    """Build C++ extension if conditions are met."""
    if not should_build_cpp_extension():
        return None

    # Check if torch is available
    try:
        from torch.utils import cpp_extension
    except ImportError:
        print("⚠️  PyTorch not found, skipping C++ extension")
        print("   Install with: pip install torch")
        return None

    # C++ source files
    cpp_sources = [
        "csrc/ops/bbox_ops.cpp",
        "csrc/bindings.cpp",
    ]

    # Check if source files exist
    for src in cpp_sources:
        if not (HERE / src).exists():
            print(f"⚠️  Source file not found: {src}")
            print("   Skipping C++ extension build")
            return None

    # Include directories
    include_dirs = [str(HERE / "include"), *cpp_extension.include_paths()]

    # Library directories
    library_dirs = cpp_extension.library_paths()

    # Libraries to link
    libraries = ["c10", "torch", "torch_cpu", "torch_python"]

    # CUDA support (optional)
    use_cuda = os.environ.get("INFERFLOW_CUDA", "0") == "1"
    if use_cuda:
        libraries.extend(["c10_cuda", "torch_cuda"])
        print("   🚀 CUDA support:  ENABLED")
    else:
        print("   ℹ️  CUDA support:  DISABLED (set INFERFLOW_CUDA=1 to enable)")

    # Compiler flags
    extra_compile_args = {"cxx": ["-O3", "-g"]}
    extra_link_args = []

    if sys.platform == "win32":
        extra_compile_args["cxx"].extend(["/std:c++17", "/MD"])
    else:
        extra_compile_args["cxx"].extend(["-std: c++17", "-Wall", "-Wextra", "-fPIC"])
        if sys.platform == "darwin":
            # macOS specific
            extra_compile_args["cxx"].append("-stdlib=libc++")
            extra_link_args.append("-stdlib=libc++")

    # Try to link against TorchVision for optimized NMS
    try:
        import torchvision

        tv_include = pathlib.Path(torchvision.__file__).parent / "include"
        if tv_include.exists():
            include_dirs.append(str(tv_include))
            print(f"   ✅ TorchVision C++ headers found:  {tv_include}")
    except ImportError:
        print("   ⚠️  TorchVision not found, using fallback NMS")

    print("   ✅ Building C++ extension:  inferflow._C")

    return cpp_extension.CppExtension(
        name="inferflow._C",
        sources=cpp_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )


if __name__ == "__main__":
    ext_module = get_cpp_extension()

    if ext_module is not None:
        from torch.utils import cpp_extension

        setup(
            ext_modules=[ext_module],
            cmdclass={"build_ext": cpp_extension.BuildExtension},
        )
    else:
        # Pure Python installation
        print("\n📦 Installing InferFlow (pure Python mode)")
        setup()
