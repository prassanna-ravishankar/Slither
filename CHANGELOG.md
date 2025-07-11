# Changelog

All notable changes to the Slither project modernization will be documented in this file.

## [2025-07-09] - Phase 1 vcpkg Integration Complete

### ✅ Completed: vcpkg Migration and Build System Modernization
- **vcpkg Integration**: Migrated from Hunter to vcpkg package manager
  - Created `vcpkg.json` manifest file with modern dependencies
  - Updated `build.sh` script to automatically detect and use vcpkg toolchain
  - Added proper VCPKG_ROOT environment variable support
- **Boost Removal**: Successfully removed boost dependencies from core build
  - Removed `boost-program-options` and `boost-serialization` from vcpkg.json
  - Updated CMakeLists.txt to use CLI11 instead of boost::program_options
  - Migrated main.cpp from boost to CLI11 argument parsing
  - Temporarily disabled boost serialization calls (marked for nlohmann/json replacement)
- **Build Infrastructure**: Enhanced build system reliability
  - Updated .gitignore to exclude build directories and vcpkg files
  - Fixed line ending issues in build scripts for cross-platform compatibility
  - Added autoconf/automake dependencies for Python3 vcpkg builds
- **Dependencies Modernized**:
  - OpenCV 4.11.0 (latest stable)
  - Eigen 3.4.0 for linear algebra
  - CLI11 2.5.0 for command-line parsing
  - nlohmann/json 3.12.0 (ready for serialization replacement)
  - pybind11 2.13.6 for Python bindings

### 🔄 In Progress: Serialization Migration
- Boost serialization calls temporarily commented out in main.cpp
- Ready for nlohmann/json implementation in next phase

## [2025-07-11] - Phase 1 Modernization Complete

### ✅ Completed: Core C++ Modernization
- **JSON Serialization**: Replaced Boost serialization with nlohmann/json
  - Implemented JSON serialization/deserialization in Forest.h, Tree.h, Node.h
  - Added JSON methods to LinearFeatureResponseSVM and HistogramAggregator
  - Enabled forest->SerializeJson() and Forest::DeserializeJson() in main.cpp
  - Forest models now saved as human-readable JSON files
- **Memory Management**: Fixed manual memory management in PlotCanvas
  - Replaced raw pointer (unsigned char*) with std::unique_ptr<unsigned char[]>
  - Removed manual delete[] in destructor
  - Updated all buffer access to use smart pointer methods
- **Build System**: Fixed compilation issues
  - Added conditional compilation for OpenMP support
  - Fixed include paths and library dependencies
  - Library builds successfully without OpenMP on macOS

## [Planned] - Remaining Modernization Roadmap

### Phase 1: Critical C++ Modernization (Continued)
- [x] Replace C-style random number generation with C++11 `<random>` library (lib/Random.h)
- [x] Convert raw pointer containers to smart pointers (lib/Forest.h)
- [x] Replace Boost Program Options with CLI11 (source/main.cpp)
- [x] Complete Boost Serialization replacement with nlohmann/json
- [x] Fix manual memory management in PlotCanvas (source/PlotCanvas.h)

### Phase 2: Python Bindings Modernization
- [ ] Redesign API to be scikit-learn compatible (SlitherClassifier class)
- [ ] Add modern build configuration (pyproject.toml)
- [ ] Fix Python module installation to link against installed library
- [ ] Add comprehensive type hints using numpy.typing
- [ ] Implement proper exception handling and custom exception hierarchy
- [ ] Add zero-copy numpy array handling with buffer protocol

### Phase 3: Performance and Quality Improvements
- [x] Add Eigen library for optimized linear algebra operations
- [x] Update OpenCV to latest 4.x series
- [ ] Consider Intel TBB for better task-based parallelism
- [ ] Replace custom CSV parsing with fast modern library
- [ ] Add comprehensive test suite with pytest
- [ ] Implement property-based testing with hypothesis

### Phase 4: Modern Development Practices
- [x] Update dependency management from Hunter to vcpkg
- [x] Modernize build system with proper vcpkg integration
- [ ] Set up CI/CD pipeline with GitHub Actions
- [ ] Generate API documentation with Sphinx (Python) and Doxygen (C++)
- [ ] Add cross-platform support (Windows)
- [ ] Implement memory-mapped file I/O for large datasets

## [Current] - Legacy State

### C++ Library
- Uses C++17 standard ✓
- Hunter package manager for dependencies (v0.23.33 from 2018)
- OpenCV integration for data handling
- Boost dependencies: serialization, program_options
- OpenMP for parallelization
- Raw pointer management in several places
- C-style random number generation

### Python Bindings
- Uses pybind11 ✓ (modern choice)
- Non-Pythonic API design (slitherWrapper class)
- Manual data copying, no zero-copy optimization
- Links against build directory instead of installed library
- Missing type hints and modern Python packaging
- Basic exception handling

### Known Issues
- Python 2/3 compatibility code in tests
- Deprecated sklearn APIs in test files
- TODO comments indicating incomplete features
- Manual memory management with new[]/delete[]
- *nix only support (no Windows)
- Build directory dependency for Python module