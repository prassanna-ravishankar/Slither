# Installation

This guide covers different ways to install Slither Random Forest.

## Requirements

### System Requirements

- **Operating System**: macOS, Linux, Windows
- **Python**: 3.8 or higher
- **C++ Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2019+)
- **CMake**: 3.16 or higher (modern CMake features required)

### Dependencies

#### Python Dependencies
- `numpy >= 1.19.0`
- `scikit-learn >= 1.0.0`

#### C++ Dependencies (auto-installed via vcpkg)
- `eigen3` - Linear algebra library
- `libsvm` - SVM implementation
- `nlohmann/json` - JSON serialization
- `cli11` - Command line parsing
- `pybind11` - Python bindings

## Installation Methods

### Method 1: PyPI (Recommended - Coming Soon)

!!! note "PyPI Release"
    PyPI packages are in preparation. For now, please use the source installation.

```bash
pip install slither-rf
```

### Method 2: Source Installation

#### Prerequisites

First, ensure you have the required tools:

=== "macOS"

    ```bash
    # Install Xcode command line tools
    xcode-select --install
    
    # Install vcpkg (if not already installed)
    git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
    ~/vcpkg/bootstrap-vcpkg.sh
    export VCPKG_ROOT=~/vcpkg
    ```

=== "Ubuntu/Debian"

    ```bash
    # Install build tools
    sudo apt update
    sudo apt install build-essential cmake git python3-dev
    
    # Install vcpkg
    git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
    ~/vcpkg/bootstrap-vcpkg.sh
    export VCPKG_ROOT=~/vcpkg
    ```

=== "CentOS/RHEL"

    ```bash
    # Install build tools
    sudo yum groupinstall "Development Tools"
    sudo yum install cmake3 git python3-devel
    
    # Install vcpkg
    git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
    ~/vcpkg/bootstrap-vcpkg.sh
    export VCPKG_ROOT=~/vcpkg
    ```

#### Clone and Build

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither

# Set up environment (add to your shell profile for persistence)
export VCPKG_ROOT=~/vcpkg

# Build using the provided script
./build.sh

# Install Python package in development mode
pip install -e .
```

#### Alternative: Manual Build

If you prefer to build manually:

```bash
# Create build directory
mkdir slither_build
cd slither_build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
make -j$(nproc)

# Install (optional)
make install

# Build Python extension
cd ..
pip install -e .
```

### Method 3: Development Installation

For developers who want to contribute:

```bash
# Clone with development dependencies
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither

# Install in development mode with extra dependencies
pip install -e ".[dev,test]"

# Set up pre-commit hooks
pre-commit install
```

## Verification

Test your installation:

```python
import slither
print(f"Slither version: {slither.__version__}")

# Quick functionality test
from slither import SlitherClassifier
import numpy as np

X = np.random.random((100, 10))
y = np.random.randint(0, 2, 100)

clf = SlitherClassifier(n_estimators=5, verbose=False)
clf.fit(X, y)
accuracy = clf.score(X, y)
print(f"Test accuracy: {accuracy:.3f}")
```

Expected output:
```
Slither version: 2.0.0
Test accuracy: 0.850
```

## Troubleshooting

### Common Issues

#### vcpkg Not Found

```bash
# Error: vcpkg toolchain not detected
export VCPKG_ROOT=/path/to/your/vcpkg
```

#### Missing Dependencies

```bash
# Install missing system dependencies
sudo apt install build-essential cmake python3-dev  # Ubuntu
brew install cmake                                   # macOS
```

#### OpenMP Issues on macOS

OpenMP is optional. If you encounter issues:

```bash
# The build will work without OpenMP
# Warning: "OpenMP not found. Building without OpenMP support."
```

#### Python Module Import Error

```bash
# Error: "Could not import the Slither C++ extension"

# Solution 1: Check PYTHONPATH
export PYTHONPATH=/path/to/Slither/slither_build:$PYTHONPATH

# Solution 2: Rebuild the extension
cd Slither
pip install -e . --force-reinstall
```

### Build Issues

#### Eigen Assertion Errors

If you see Eigen-related assertion errors during testing:

```bash
# This is usually due to data format issues
# Try with smaller datasets or different parameters
clf = SlitherClassifier(
    n_estimators=3,     # Fewer trees
    max_depth=3,        # Shallower trees  
    svm_c=0.1          # Smaller SVM regularization
)
```

#### Memory Issues

For large datasets:

```bash
# Reduce memory usage
clf = SlitherClassifier(
    n_estimators=10,           # Fewer trees
    n_candidate_features=10,   # Fewer features per split
    n_jobs=1                   # Single threaded
)
```

## Platform-Specific Notes

### macOS

- Xcode command line tools are required
- OpenMP is not available by default (optional dependency)
- Apple Silicon (M1/M2) is fully supported

### Linux

- Most distributions work out of the box
- OpenMP is usually available
- Both x86_64 and ARM64 are supported

### Windows

Windows support is planned for future releases. Current limitations:

- MSVC compilation needs testing
- vcpkg integration may need adjustments
- Some path handling issues in build scripts

## Docker Installation

For containerized environments:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN /opt/vcpkg/bootstrap-vcpkg.sh
ENV VCPKG_ROOT=/opt/vcpkg

# Install Slither
COPY . /app
WORKDIR /app
RUN ./build.sh && pip install -e .
```

## Next Steps

Once installed, proceed to:

- [Quick Start Guide](quickstart.md) - Learn basic usage
- [Examples](examples.md) - See practical applications
- [API Reference](../api/python.md) - Detailed API documentation