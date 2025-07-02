# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slither is a Random Forest library implementing local SVM experts at tree nodes, designed for computer vision tasks. It was developed for road segmentation using hypercolumn features and implements Information Gain-based splitting (not Gini criteria). The library supports classification, regression, density estimation, and semi-supervised learning.

## Build Commands

### C++ Library and Executable
```bash
mkdir slither_build
cd slither_build
cmake ../Slither
make -j && make install
```

### Python Library (automatically builds C++ library)
```bash
python setup.py install
```
Note: Python module links against libSlither in the build directory - don't delete the build folder after installation.

### Testing
```bash
# Python tests (requires scikit-learn)
python test/test_pyslither.py

# C++ executable (self-explanatory arguments)
./cppSlither --help
```

## Architecture Overview

### Core Library (`/lib/`)
- **Sherwood.h**: Main decision forest framework header inspired by Microsoft's Sherwood
- **Forest.h, Tree.h, Node.h**: Core data structures for random forest implementation
- **ForestTrainer.h, ParallelForestTrainer.h**: Training implementations with OpenMP support
- **Interfaces.h**: Abstract interfaces for extensibility
- **TrainingParameters.h**: Configuration management

### Implementation (`/source/`)
- **Classification.cpp/h, Regression.cpp/h**: ML task implementations
- **FeatureResponseFunctions.cpp/h**: SVM-based feature response functions at tree nodes
- **StatisticsAggregators.cpp/h**: Node statistics aggregation
- **DataPointCollection.cpp/h**: OpenCV-based data management
- **main.cpp**: Command-line interface using Boost program_options

### Python Bindings (`/pyWrapper/`)
- **wrapper.cpp**: pybind11-based Python interface exposing slitherWrapper class
- Handles numpy arrays but requires manual data copying (no zero-copy optimization)

### Data Format
- Uses OpenCV Mat internally for data representation
- Supports tab-delimited text files for training data
- Built-in serialization using Boost::serialization

## Key Dependencies
- **OpenCV**: Must be pre-installed (not auto-downloaded by Hunter)
- **Boost**: serialization and program_options (auto-downloaded via Hunter)
- **pybind11**: For Python bindings (auto-downloaded)
- **C++17 compiler**: Required standard
- **Python dev libraries**: For building Python bindings

## Modernization Plan

**Note**: Progress is tracked in `CHANGELOG.md` - update it when completing modernization tasks.

### Phase 1: Critical C++ Modernization (High Priority)
1. **Replace C-style random number generation** (lib/Random.h)
   - Replace `rand()`, `srand()`, `time(NULL)` with C++11 `<random>` library
   - Use `std::mt19937` and proper distributions for thread-safety

2. **Convert raw pointer containers to smart pointers** (lib/Forest.h)
   - Replace `std::vector<Tree<F,S>*>` with `std::vector<std::unique_ptr<Tree<F,S>>>`
   - Eliminate manual memory management in destructors

3. **Replace Boost Program Options** (source/main.cpp)
   - Migrate to CLI11 or cxxopts for modern command-line parsing
   - Remove dependency on Boost program_options

4. **Replace Boost Serialization**
   - Migrate to nlohmann/json for human-readable format or Protocol Buffers for performance
   - Reduce Boost dependencies

### Phase 2: Python Bindings Modernization (High Priority)
1. **API Redesign for Python Standards**
   - Replace `slitherWrapper` with scikit-learn compatible `SlitherClassifier`
   - Implement standard methods: `fit()`, `predict()`, `predict_proba()`, `score()`
   - Remove non-Pythonic method names like `onlyTrain()`, `onlyTest()`

2. **Modern Build Configuration**
   - Add `pyproject.toml` with proper build-system specification
   - Add `python_requires=">=3.8"`
   - Fix installation to link against installed library, not build directory

3. **Type Hints and Error Handling**
   - Add complete type annotations using `numpy.typing`
   - Create custom exception hierarchy
   - Map C++ exceptions to appropriate Python exceptions

### Phase 3: Performance and Quality (Medium Priority)
1. **Add Eigen for linear algebra** alongside OpenCV
2. **Update OpenCV to 4.x** series for better performance
3. **Consider Intel TBB** instead of OpenMP for better task parallelism
4. **Replace custom CSV parsing** with fast modern CSV library
5. **Zero-copy numpy array handling** in Python bindings

### Phase 4: Modern Development Practices (Low Priority)
1. **Update dependency management** from Hunter to Conan/vcpkg
2. **Add comprehensive test suite** with pytest and property-based testing
3. **Set up CI/CD pipeline** with GitHub Actions
4. **Generate API documentation** with Sphinx/Doxygen

## Development Notes

### Current Issues
- Python module installation links to build directory (not proper installation)
- Uses deprecated random number generation (lib/Random.h uses rand/srand)
- Raw pointer management in Forest class stores Tree*
- Several TODO comments indicate incomplete features
- Only supports *nix systems currently
- Works best with Python 3 in virtual environments

### Code Patterns
- Template-heavy design following Microsoft Sherwood patterns
- Heavy use of OpenCV data structures throughout
- C-style patterns mixed with modern C++17

### Data Organization
Sample datasets are provided in `/data/` for different ML tasks:
- `supervised classification/`: Binary and multi-class data
- `semi-supervised classification/`: Mixed labeled/unlabeled data  
- `regression/`: Regression examples
- `density estimation/`: Density estimation data
- `sclf/`: Special classification data with Python utilities