# Building from Source

Complete guide for building Slither Random Forest from source code, including all dependencies and development setup.

## Prerequisites

### System Requirements

| Platform | Minimum | Recommended |
|----------|---------|-------------|
| **macOS** | 10.14+ | 12.0+ (Monterey) |
| **Linux** | Ubuntu 18.04+, CentOS 7+ | Ubuntu 22.04+, CentOS 9+ |
| **Windows** | Windows 10+ | Windows 11+ (experimental) |

### Required Tools

#### Compiler Support

```bash
# GCC (Linux)
gcc --version  # Requires GCC 7.0+
g++ --version  # C++17 support required

# Clang (macOS/Linux) 
clang --version  # Requires Clang 5.0+
clang++ --version

# MSVC (Windows)
# Requires Visual Studio 2019+ or Build Tools
```

#### Build Tools

```bash
# CMake (required)
cmake --version  # Requires CMake 3.8+

# Make/Ninja
make --version   # Traditional build system
ninja --version  # Faster alternative

# Git
git --version    # For cloning repositories

# Python (for bindings)
python3 --version  # Requires Python 3.8+
```

## Dependency Management

### vcpkg Setup

Slither uses vcpkg for C++ dependency management:

#### Installation

=== "macOS/Linux"

    ```bash
    # Clone vcpkg
    git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
    
    # Bootstrap
    ~/vcpkg/bootstrap-vcpkg.sh
    
    # Add to environment
    export VCPKG_ROOT=~/vcpkg
    export PATH=$VCPKG_ROOT:$PATH
    
    # Make permanent (add to ~/.bashrc or ~/.zshrc)
    echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
    echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
    ```

=== "Windows"

    ```cmd
    REM Clone vcpkg
    git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
    
    REM Bootstrap
    C:\vcpkg\bootstrap-vcpkg.bat
    
    REM Set environment variable (permanent)
    setx VCPKG_ROOT "C:\vcpkg"
    ```

#### Dependency Installation

```bash
# Navigate to vcpkg directory
cd $VCPKG_ROOT

# Install Slither dependencies
./vcpkg install eigen3
./vcpkg install libsvm  
./vcpkg install nlohmann-json
./vcpkg install cli11
./vcpkg install pybind11

# Optional: Install all dependencies for your platform
./vcpkg install eigen3:x64-linux      # Linux
./vcpkg install eigen3:x64-osx        # macOS
./vcpkg install eigen3:x64-windows    # Windows
```

### Alternative: System Package Managers

If you prefer system packages over vcpkg:

=== "Ubuntu/Debian"

    ```bash
    # Update package list
    sudo apt update
    
    # Install build dependencies
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        python3-dev \
        python3-pip \
        libeigen3-dev \
        libsvm-dev \
        nlohmann-json3-dev
    
    # Install Python dependencies
    pip3 install pybind11[global] numpy scikit-learn
    ```

=== "CentOS/RHEL"

    ```bash
    # Install EPEL repository
    sudo yum install -y epel-release
    
    # Install build dependencies
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake3 \
        git \
        python3-devel \
        python3-pip \
        eigen3-devel
    
    # Install Python dependencies
    pip3 install pybind11[global] numpy scikit-learn
    ```

=== "macOS (Homebrew)"

    ```bash
    # Install Homebrew if not present
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install dependencies
    brew install cmake git python@3.11 eigen libsvm nlohmann-json
    
    # Install Python dependencies
    pip3 install pybind11[global] numpy scikit-learn
    ```

## Building the Library

### Quick Build

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither

# Run build script (recommended)
./build.sh

# Install Python package
pip install -e .
```

### Manual Build Process

#### Step 1: Configure

```bash
# Create build directory
mkdir slither_build
cd slither_build

# Configure with CMake
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_OPENMP=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=ON
```

#### Step 2: Build

```bash
# Build with make
make -j$(nproc)

# Alternative: Build with ninja (faster)
ninja
```

#### Step 3: Install

```bash
# Install C++ library (optional)
sudo make install

# Install Python package
cd ..
pip install -e .
```

### Build Options

#### CMake Configuration Options

```bash
# Build type options
-DCMAKE_BUILD_TYPE=Release    # Optimized build (default)
-DCMAKE_BUILD_TYPE=Debug      # Debug symbols
-DCMAKE_BUILD_TYPE=RelWithDebInfo  # Optimized + debug info

# Feature options
-DWITH_OPENMP=ON             # Enable OpenMP parallelization
-DWITH_OPENMP=OFF            # Disable OpenMP (fallback)
-DBUILD_TESTS=ON             # Build C++ tests
-DBUILD_PYTHON_BINDINGS=ON   # Build Python wrapper
-DBUILD_EXAMPLES=ON          # Build example programs

# Installation options  
-DCMAKE_INSTALL_PREFIX=/usr/local  # Installation directory
-DCMAKE_INSTALL_LIBDIR=lib64       # Library directory (for 64-bit)

# Compiler options
-DCMAKE_CXX_COMPILER=g++           # Specify compiler
-DCMAKE_CXX_FLAGS="-O3 -march=native"  # Custom flags
```

#### Example Configurations

```bash
# Development build
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTS=ON \
    -DWITH_OPENMP=OFF \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Production build  
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_OPENMP=ON \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native" \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Python-only build
cmake .. \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=OFF \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

## Python Extension Building

### Development Installation

```bash
# Install in development mode (recommended for development)
pip install -e .

# Install with verbose output
pip install -e . -v

# Force rebuild
pip install -e . --force-reinstall

# Install specific extras
pip install -e ".[dev,test]"
```

### setuptools Configuration

The `setup.py` uses CMake build extension:

```python
# setup.py key components
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "slitherWrapper",
        sorted(glob.glob("pyWrapper/*.cpp")),
        include_dirs=[
            "lib/",
            "source/", 
            get_pybind_include(),
        ],
        libraries=["slither"],
        library_dirs=["slither_build/"],
        language='c++',
        cxx_std=17,
    ),
]
```

### pyproject.toml Build System

```toml
[build-system]
requires = [
    "setuptools>=45",
    "wheel",
    "pybind11>=2.10.0",
    "cmake>=3.12",
]
build-backend = "setuptools.build_meta"
```

## Testing the Build

### C++ Tests

```bash
# From build directory
cd slither_build

# Run C++ executable tests
./slither_cpp --help

# Test with sample data
./slither_cpp \
    --train ../data/supervised\ classification/binary_classification.txt \
    --test ../data/supervised\ classification/binary_classification.txt \
    --trees 5 \
    --depth 3
```

### Python Tests

```bash
# Run Python test suite
cd Slither
python test/test_pyslither.py

# Quick functionality test
python -c "
from slither import SlitherClassifier
import numpy as np
X = np.random.random((100, 10))
y = np.random.randint(0, 2, 100)
clf = SlitherClassifier(n_estimators=3, verbose=False)
clf.fit(X, y)
print(f'Test accuracy: {clf.score(X, y):.3f}')
"
```

### Build Verification

```bash
# Verify C++ library
ls -la slither_build/libslither*

# Verify Python module
python -c "import slitherWrapper; print('C++ extension loaded successfully')"
python -c "from slither import SlitherClassifier; print('Python package loaded successfully')"

# Check library dependencies (Linux)
ldd slither_build/libslither.so

# Check library dependencies (macOS)  
otool -L slither_build/libslither.dylib
```

## Cross-Platform Considerations

### macOS Specifics

#### Apple Silicon (M1/M2) Support

```bash
# Check architecture
uname -m  # Should show arm64

# Build for Apple Silicon
cmake .. \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Universal binary (Intel + Apple Silicon)
cmake .. \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

#### Xcode Integration

```bash
# Generate Xcode project
cmake .. -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Open in Xcode
open Slither.xcodeproj
```

#### OpenMP Issues

```bash
# OpenMP may not be available on macOS
# Build will continue with warning:
# "OpenMP not found. Building without OpenMP support."

# To install OpenMP (optional):
brew install libomp

# Then rebuild with:
cmake .. \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib
```

### Linux Distribution Specifics

#### Ubuntu/Debian

```bash
# Install build dependencies
sudo apt install -y build-essential cmake git python3-dev

# Fix potential locale issues
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# For older Ubuntu versions, use cmake3
sudo apt install cmake3
alias cmake=cmake3
```

#### CentOS/RHEL

```bash
# Enable devtoolset for newer GCC
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++

# Activate devtoolset
scl enable devtoolset-8 bash

# Use cmake3 instead of cmake
sudo yum install -y cmake3
alias cmake=cmake3
```

#### Arch Linux

```bash
# Install dependencies
sudo pacman -S base-devel cmake git python python-pip eigen

# Install from AUR if needed
yay -S libsvm
```

### Windows Specifics (Experimental)

#### Visual Studio Setup

```cmd
REM Install Visual Studio 2019 or later with C++ workload

REM Configure environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

REM Build
cmake --build . --config Release
```

## Troubleshooting

### Common Build Issues

#### CMake Cannot Find vcpkg

```bash
# Error: vcpkg toolchain not found
export VCPKG_ROOT=/path/to/your/vcpkg
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

#### Missing Dependencies

```bash
# Error: Could not find eigen3, libsvm, etc.
cd $VCPKG_ROOT
./vcpkg install eigen3 libsvm nlohmann-json cli11 pybind11

# Clear CMake cache and reconfigure
rm -rf CMakeCache.txt CMakeFiles/
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

#### Python Extension Import Error

```bash
# Error: ImportError: cannot import name 'slitherWrapper'

# Check if extension was built
ls slither_build/slitherWrapper*

# Rebuild Python extension
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add build directory to Python path
export PYTHONPATH=/path/to/Slither/slither_build:$PYTHONPATH
```

#### OpenMP Linking Issues

```bash
# Error: undefined reference to 'omp_get_thread_num'

# Disable OpenMP
cmake .. -DWITH_OPENMP=OFF

# Or install OpenMP development package
sudo apt install libomp-dev  # Ubuntu
brew install libomp          # macOS
```

#### C++17 Support Issues

```bash
# Error: C++17 features not supported

# Check compiler version
g++ --version  # Needs GCC 7+
clang++ --version  # Needs Clang 5+

# Upgrade compiler if needed
sudo apt install gcc-9 g++-9  # Ubuntu
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
```

### Memory Issues During Build

```bash
# Reduce parallel jobs if running out of memory
make -j2  # Instead of make -j$(nproc)

# Or use swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Issues

```bash
# Use ninja instead of make (faster)
cmake .. -G Ninja
ninja

# Use ccache for faster rebuilds
sudo apt install ccache
export CXX="ccache g++"
export CC="ccache gcc"
```

## Custom Build Configurations

### Development Build Script

Create a custom build script for development:

```bash
#!/bin/bash
# dev_build.sh

set -e

# Configuration
BUILD_TYPE=${1:-Debug}
BUILD_DIR="build_${BUILD_TYPE,,}"
JOBS=${2:-$(nproc)}

echo "Building Slither (${BUILD_TYPE}) with ${JOBS} jobs..."

# Clean previous build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DWITH_OPENMP=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
make -j${JOBS}

# Install Python package
cd ..
pip install -e . --force-reinstall

echo "Build complete! Run tests with:"
echo "  cd ${BUILD_DIR} && ./slither_cpp --help"
echo "  python test/test_pyslither.py"
```

### Production Build Script

```bash
#!/bin/bash
# prod_build.sh

set -e

BUILD_DIR="build_release"
PREFIX=${1:-/usr/local}

echo "Building Slither for production..."

# Clean and create build directory
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure for production
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DWITH_OPENMP=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native" \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build with all available cores
make -j$(nproc)

# Install system-wide (requires sudo)
echo "Installing to ${PREFIX} (may require sudo)..."
sudo make install

# Install Python package
cd ..
pip install .

echo "Production build complete!"
echo "Library installed to: ${PREFIX}"
```

This comprehensive building guide covers all aspects of compiling Slither from source across different platforms and configurations.