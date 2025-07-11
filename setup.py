"""
Modern setup.py for Slither Random Forest library
==================================================

This setup.py works with pyproject.toml to build the C++ extension
and Python package using modern Python packaging standards.
"""

import os
import sys
import subprocess
from pathlib import Path

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages


# Define the extension module
ext_modules = [
    Pybind11Extension(
        "slither_py",
        sources=[
            "pyWrapper/wrapper.cpp",
            # Add all source files from the C++ library
            "source/Classification.cpp",
            "source/CommandLineParser.cpp", 
            "source/CumulativeNormalDistribution.cpp",
            "source/DataPointCollection.cpp",
            "source/dibCodec.cpp",
            "source/FeatureResponseFunctions.cpp",
            "source/FloydWarshall.cpp",
            "source/Graphics.cpp",
            "source/Platform.cpp",
            "source/PlotCanvas.cpp",
            "source/StatisticsAggregators.cpp",
        ],
        include_dirs=[
            "lib",
            "source",
            get_cmake_dir(),
        ],
        language="c++",
        cxx_std=17,
    ),
]


class CMakeBuild(build_ext):
    """Custom build extension that uses CMake to build the C++ library."""
    
    def build_extension(self, ext):
        """Build the extension using CMake."""
        # Create build directory
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure with CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={Path(self.build_lib).absolute()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        
        # Add vcpkg toolchain if available
        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if vcpkg_root:
            cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        
        # Configure
        subprocess.check_call([
            "cmake", str(Path(__file__).parent.absolute()), *cmake_args
        ], cwd=build_dir)
        
        # Build
        subprocess.check_call([
            "cmake", "--build", ".", "--target", "slither_py"
        ], cwd=build_dir)


# Use the modern setup with pyproject.toml configuration
if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
    )