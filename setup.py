import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.0.0"

cpp_sources = [
    "cpp/src/foveated_grid.cpp",
    "cpp/src/bindings.cpp",
]

ext_modules = [
    Pybind11Extension(
        "foveated_grid_cpp",
        cpp_sources,
        include_dirs=["cpp/include"],
        cxx_std=17,
        extra_compile_args=["-O3", "-Wall", "-Wextra"],
    ),
]

setup(
    name="foveated_lidar_mapping",
    version=__version__,
    author="Amit Kumar Tripathi",
    description="High-Performance Foveated 2.5D LiDAR Mapping Pipeline with SPVCNN and C++ Grid Engine",
    packages=find_packages(include=["src", "src.*", "phase2", "phase2.*", "ml", "ml.*", "visualization", "visualization.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
