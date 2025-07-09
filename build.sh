#!/bin/bash

# Build script for Slither using vcpkg for dependencies

echo "Slither Build Script"
echo "==================="

# Check if vcpkg toolchain is provided
if [ ! -z "$1" ] && [ -f "$1" ]; then
    echo "Using vcpkg toolchain: $1"
    TOOLCHAIN_ARGS="-DCMAKE_TOOLCHAIN_FILE=$1"
elif [ ! -z "$CMAKE_TOOLCHAIN_FILE" ] && [ -f "$CMAKE_TOOLCHAIN_FILE" ]; then
    echo "Using vcpkg toolchain from environment: $CMAKE_TOOLCHAIN_FILE"
    TOOLCHAIN_ARGS="-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE"
elif [ ! -z "$VCPKG_ROOT" ] && [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
    echo "Using vcpkg toolchain from VCPKG_ROOT: $VCPKG_ROOT"
    TOOLCHAIN_ARGS="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
else
    echo "WARNING: No vcpkg toolchain detected. Build may fail if dependencies are not available."
    echo "To use vcpkg, either:"
    echo "  1. Pass the toolchain file as argument: ./build.sh /path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
    echo "  2. Set CMAKE_TOOLCHAIN_FILE environment variable"
    echo "  3. Set VCPKG_ROOT environment variable"
    echo ""
    echo "Attempting to build with system dependencies..."
    TOOLCHAIN_ARGS=""
fi

# Create build directory
mkdir -p slither_build
cd slither_build

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. $TOOLCHAIN_ARGS

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Install
echo ""
echo "Installing..."
make install

echo ""
echo "Build complete!"