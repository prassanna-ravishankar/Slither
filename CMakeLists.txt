cmake_minimum_required(VERSION 3.16)

# Project definition
project(Slither
    VERSION 1.0.0
    DESCRIPTION "Random Forest library with SVM local experts for computer vision"
    LANGUAGES CXX)

# Check if using vcpkg
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARNING "vcpkg toolchain not detected. Please ensure vcpkg is properly integrated.")
endif()

# Modern CMake settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build configuration
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(SLITHER_BUILD_TESTS "Build tests" ON)
option(SLITHER_BUILD_EXAMPLES "Build examples" ON)
option(SLITHER_BUILD_BENCHMARKS "Build benchmarks" OFF)

#######################################
# Dependencies
#######################################

# Required dependencies
find_package(unofficial-libsvm CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json CONFIG REQUIRED)

# Optional dependencies
find_package(OpenMP)
find_package(CLI11 CONFIG REQUIRED)

# For Python bindings
find_package(pybind11 CONFIG QUIET)

# Print found library versions
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "CLI11 found: ${CLI11_FOUND}")
message(STATUS "nlohmann_json found: ${nlohmann_json_FOUND}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")

#######################################
# Library source files
#######################################

set(SLITHER_SOURCES
    src/Classification.cpp
    src/CommandLineParser.cpp
    src/CumulativeNormalDistribution.cpp
    src/DataPointCollection.cpp
    src/DensityEstimation.cpp
    src/FeatureResponseFunctions.cpp
    src/FloydWarshall.cpp
    src/Graphics.cpp
    src/Platform.cpp
    src/PlotCanvas.cpp
    src/Regression.cpp
    src/SemiSupervisedClassification.cpp
    src/StatisticsAggregators.cpp
)

set(SLITHER_HEADERS
    include/slither/Forest.h
    include/slither/Tree.h
    include/slither/Node.h
    include/slither/ForestTrainer.h
    include/slither/ParallelForestTrainer.h
    include/slither/Interfaces.h
    include/slither/Random.h
    include/slither/Sherwood.h
    include/slither/TrainingParameters.h
    include/slither/ProgressStream.h
    include/slither/Classification.h
    include/slither/Regression.h
    include/slither/DensityEstimation.h
    include/slither/SemiSupervisedClassification.h
    include/slither/FeatureResponseFunctions.h
    include/slither/StatisticsAggregators.h
    include/slither/DataPointCollection.h
    include/slither/Graphics.h
    include/slither/Platform.h
)

#######################################
# Main library
#######################################

add_library(Slither ${SLITHER_SOURCES} ${SLITHER_HEADERS})
add_library(Slither::Slither ALIAS Slither)

# Include directories
target_include_directories(Slither
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(Slither
    PUBLIC
        unofficial::libsvm::libsvm
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# OpenMP support
if(OpenMP_FOUND)
    target_link_libraries(Slither PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(Slither PUBLIC WITH_OPENMP)
    message(STATUS "OpenMP support enabled")
else()
    message(STATUS "OpenMP not found. Building without parallel support.")
    find_package(Threads REQUIRED)
    target_link_libraries(Slither PUBLIC Threads::Threads)
endif()

# Compiler-specific options
target_compile_features(Slither PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(Slither PRIVATE /W4)
else()
    target_compile_options(Slither PRIVATE -Wall -Wextra -Wpedantic)
endif()

#######################################
# Executable
#######################################

if(SLITHER_BUILD_EXAMPLES)
    add_executable(slither_cpp src/main.cpp)
    target_link_libraries(slither_cpp 
        PRIVATE 
            Slither::Slither
            CLI11::CLI11
    )
    
    # Install executable
    install(TARGETS slither_cpp
        RUNTIME DESTINATION bin
    )
endif()

#######################################
# Python bindings
#######################################

if(pybind11_FOUND)
    pybind11_add_module(slither_py pyWrapper/wrapper.cpp)
    target_link_libraries(slither_py PRIVATE Slither::Slither)
    
    # Install Python module
    install(TARGETS slither_py
        LIBRARY DESTINATION python/slither
    )
endif()

#######################################
# Tests
#######################################

if(SLITHER_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_datapointcollection 
        tests/test_datapointcollection.cpp
    )
    target_link_libraries(test_datapointcollection 
        PRIVATE 
            Slither::Slither
    )
    
    add_test(NAME DataPointCollectionTest COMMAND test_datapointcollection)
endif()

#######################################
# Benchmarks
#######################################

if(SLITHER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

#######################################
# Installation
#######################################

# Install library
install(TARGETS Slither
    EXPORT SlitherTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Generate and install CMake config files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    SlitherConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT SlitherTargets
    FILE SlitherTargets.cmake
    NAMESPACE Slither::
    DESTINATION lib/cmake/Slither
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SlitherConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SlitherConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Slither
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SlitherConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SlitherConfigVersion.cmake"
    DESTINATION lib/cmake/Slither
)

# Export targets for build tree
export(EXPORT SlitherTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SlitherTargets.cmake"
    NAMESPACE Slither::
)