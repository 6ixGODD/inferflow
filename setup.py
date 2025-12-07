#!/usr/bin/env python

import os
import pathlib
import sys

from setuptools import setup
from torch.utils import cpp_extension

HERE = pathlib.Path(__file__).parent.resolve()


def get_cpp_extension():
    # C++ source files
    cpp_sources = [
        "csrc/ops/bbox_ops.cpp",
        "csrc/bindings.cpp",
    ]

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
        print("🚀 Building with CUDA support")

    # Compiler flags
    extra_compile_args = {"cxx": ["-O3", "-g"]}
    extra_link_args = []

    if sys.platform == "win32":
        extra_compile_args["cxx"].extend(["/std:c++17", "/MD"])
    else:
        extra_compile_args["cxx"].extend(["-std=c++17", "-Wall", "-Wextra", "-fPIC"])
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
            print(f"✅ Found TorchVision C++ headers at {tv_include}")
    except ImportError:
        torchvision = None
        print("⚠️  TorchVision not found, using fallback NMS implementation")

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
    setup(
        ext_modules=[get_cpp_extension()],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
