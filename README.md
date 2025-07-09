# 🐍 Slither

> A high-performance Random Forest library with SVM local experts for computer vision tasks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![vcpkg](https://img.shields.io/badge/vcpkg-ready-green.svg)](https://vcpkg.io)

## 📖 Overview

Slither is a specialized Random Forest implementation designed for computer vision tasks, featuring **SVM local experts** at tree nodes. Originally developed for road segmentation using hypercolumn features, it implements **Information Gain-based splitting** (not Gini criteria) and supports classification, regression, density estimation, and semi-supervised learning.

### 🔬 Research Background

This is the official implementation for the paper:
**[Unstructured Road Segmentation using Hypercolumn based Random Forests of Local experts](https://figshare.com/articles/Unstructured_Road_Segmentation_using_Hypercolumn_based_Random_Forests_of_Local_experts/7241360)**

### ✨ Key Features

- **🧠 SVM Local Experts**: Support Vector Machines at tree nodes for enhanced decision boundaries
- **📊 Information Gain Splitting**: More sophisticated splitting criteria than traditional Gini
- **🎯 Computer Vision Optimized**: Designed for hypercolumn features and image processing
- **🔄 Multiple Learning Modes**: Classification, regression, density estimation, semi-supervised
- **⚡ High Performance**: OpenMP parallelization and optimized linear algebra
- **🐍 Python Bindings**: Easy-to-use Python interface with pybind11

### 🏗️ Architecture

Inspired by Microsoft Cambridge Research's [Sherwood library](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52340), Slither modernizes the approach with:
- OpenCV-native data handling
- Modern C++17 implementation
- vcpkg dependency management
- JSON-based serialization

## 🚀 Quick Start

### Prerequisites

| Component | Version | Installation |
|-----------|---------|-------------|
| **C++ Compiler** | C++17 compatible | `sudo apt-get install build-essential` |
| **CMake** | 3.16+ | `sudo apt-get install cmake` |
| **vcpkg** | Latest | [Installation Guide](https://vcpkg.io/en/getting-started.html) |
| **OpenCV** | 4.x (recommended) | Pre-installed or via vcpkg |

### 🔧 Installation

#### Option 1: One-Line Build (Recommended)

```bash
# Clone and build automatically
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither
./build.sh
```

#### Option 2: Manual vcpkg Build

```bash
# 1. Setup vcpkg (if not already done)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
export VCPKG_ROOT=$(pwd)

# 2. Build Slither
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
make -j$(nproc)
```

#### Option 3: Python Package

```bash
# Install in virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy scikit-learn

# Build and install
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither
python setup.py install
```

## 💡 Usage

### C++ Command Line Interface

```bash
# Basic usage
./slither_cpp --help

# Train a model
./slither_cpp --train data/train.txt --model forest.json --trees 100 --depth 15

# Test a model
./slither_cpp --test data/test.txt --model forest.json --predict predictions.txt

# Combined training and testing
./slither_cpp --train data/train.txt --test data/test.txt --model forest.json --op_mode tr-te
```

### Python API

```python
import numpy as np
from slither_py import SlitherWrapper

# Create and train model
model = SlitherWrapper()
model.train(X_train, y_train, num_trees=100, max_depth=15)

# Make predictions
predictions = model.predict(X_test)

# Save/load model
model.save("model.json")
model.load("model.json")
```

For complete examples, see [`test/test_pyslither.py`](test/test_pyslither.py).

## 🔧 Configuration

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trees` | Number of trees in forest | 10 |
| `--depth` | Maximum decision levels | 15 |
| `--feats` | Candidate features per split | 10 |
| `--thresh` | Threshold samples per feature | 10 |
| `--svm_c` | SVM regularization parameter | 0.5 |
| `--threads` | Parallel threads | 1 |
| `--scale` | Scale input data | false |

### Data Format

Slither expects tab-delimited text files:
```
feature1    feature2    feature3    ...    label
0.1         0.2         0.3         ...    1
0.4         0.5         0.6         ...    0
...
```

## 🏗️ Dependencies

Automatically managed via vcpkg:

| Library | Purpose | Version |
|---------|---------|---------|
| **OpenCV** | Computer vision & data handling | 4.11.0 |
| **Eigen3** | Linear algebra operations | 3.4.0 |
| **CLI11** | Command line parsing | 2.5.0 |
| **nlohmann/json** | JSON serialization | 3.12.0 |
| **pybind11** | Python bindings | 2.13.6 |

## 📁 Project Structure

```
Slither/
├── 📁 lib/           # Core library headers
│   ├── Forest.h      # Random forest implementation
│   ├── Tree.h        # Decision tree structure
│   └── Node.h        # Tree node with SVM experts
├── 📁 source/        # Implementation files
│   ├── main.cpp      # CLI application
│   └── *.cpp         # Core algorithms
├── 📁 pyWrapper/     # Python bindings
├── 📁 data/          # Example datasets
├── 📁 test/          # Test files
└── 📄 build.sh       # Build script
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
./build.sh
python test/test_pyslither.py

# Submit pull request
```

## 📋 Roadmap

- [ ] **Cross-platform support** (Windows, macOS)
- [ ] **Intel TBB integration** for better parallelization
- [ ] **Scikit-learn compatible API** for Python
- [ ] **CI/CD pipeline** with GitHub Actions
- [ ] **Comprehensive documentation** with Sphinx
- [ ] **Memory-mapped file I/O** for large datasets
- [ ] **Neural network nodes** (experimental)

## 🐛 Known Issues

- **Python 3 only**: Python 2 support has been dropped
- **Unix systems only**: Windows support planned
- **Build directory dependency**: Python module links to build directory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Cambridge Research** for the original Sherwood library
- **OpenCV community** for computer vision tools
- **vcpkg team** for modern dependency management
- **Contributors** who helped modernize this codebase

## 📞 Support

- **📧 Issues**: [GitHub Issues](https://github.com/prassanna-ravishankar/Slither/issues)
- **📖 Documentation**: [Wiki](https://github.com/prassanna-ravishankar/Slither/wiki)
- **💬 Discussions**: [GitHub Discussions](https://github.com/prassanna-ravishankar/Slither/discussions)

---

<div align="center">
  <sub>Built with ❤️ for the computer vision community</sub>
</div>