# Changelog

All notable changes to the Slither project modernization will be documented in this file.

## [Planned] - Modernization Roadmap

### Phase 1: Critical C++ Modernization
- [x] Replace C-style random number generation with C++11 `<random>` library (lib/Random.h)
- [ ] Convert raw pointer containers to smart pointers (lib/Forest.h)
- [ ] Replace Boost Program Options with CLI11 or cxxopts (source/main.cpp)
- [ ] Replace Boost Serialization with nlohmann/json or Protocol Buffers
- [ ] Fix manual memory management in PlotCanvas (source/PlotCanvas.h)

### Phase 2: Python Bindings Modernization
- [ ] Redesign API to be scikit-learn compatible (SlitherClassifier class)
- [ ] Add modern build configuration (pyproject.toml)
- [ ] Fix Python module installation to link against installed library
- [ ] Add comprehensive type hints using numpy.typing
- [ ] Implement proper exception handling and custom exception hierarchy
- [ ] Add zero-copy numpy array handling with buffer protocol

### Phase 3: Performance and Quality Improvements
- [ ] Add Eigen library for optimized linear algebra operations
- [ ] Update OpenCV to latest 4.x series
- [ ] Consider Intel TBB for better task-based parallelism
- [ ] Replace custom CSV parsing with fast modern library
- [ ] Add comprehensive test suite with pytest
- [ ] Implement property-based testing with hypothesis

### Phase 4: Modern Development Practices
- [ ] Update dependency management from Hunter to Conan/vcpkg
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