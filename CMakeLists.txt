cmake_minimum_required(VERSION 3.8)

# Check if using vcpkg
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARNING "vcpkg toolchain not detected. Please ensure vcpkg is properly integrated.")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(randomSlither)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#######################################
# Dependencies for the main library
#######################################
# Build configuration
set(BUILD_SHARED_LIBS ON)

# Required : OpenCV (preferably 4.x)
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

# Optional OpenMP
find_package(OpenMP)

# Modern libraries for future modernization
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# For the library
set(SOURCE_FILES
        source/Classification.cpp
        source/CommandLineParser.cpp
        source/CumulativeNormalDistribution.cpp
        source/DataPointCollection.cpp
        #demo/source/DensityEstimation.cpp
        source/dibCodec.cpp
        source/FeatureResponseFunctions.cpp
        source/FloydWarshall.cpp
        source/Graphics.cpp
        source/Platform.cpp
        source/PlotCanvas.cpp
        #demo/source/Regression.cpp
        #demo/source/SemiSupervisedClassification.cpp
        source/StatisticsAggregators.cpp)

# Print found library versions for debugging
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "CLI11 found: ${CLI11_FOUND}")
message(STATUS "nlohmann_json found: ${nlohmann_json_FOUND}")


#######################################
# Build the library
#######################################
add_library(Slither STATIC ${SOURCE_FILES})

# Add related includes
target_include_directories(Slither PUBLIC lib)
target_include_directories(Slither PUBLIC source)

# Link related libs
target_link_libraries(Slither PUBLIC ${OpenCV_LIBS})
target_link_libraries(Slither PUBLIC nlohmann_json::nlohmann_json)
set_target_properties(Slither PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

# Adding definitions for OPENMP
if(NOT TARGET OpenMP::OpenMP_CXX)
    message("Linking OpenMP stuff with old school style")
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
else()
    message("Linking OpenMP stuff the new way")
endif()
target_link_libraries(Slither PUBLIC OpenMP::OpenMP_CXX)


#######################################
# Build the Standalone executable
#######################################
add_executable(cppSlither source/main.cpp)
# Linking and including
target_link_libraries(cppSlither PUBLIC Slither)
target_link_libraries(cppSlither PUBLIC CLI11::CLI11)

set_target_properties(cppSlither PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )


#######################################
# Build the Python Wrapper
#######################################
# First --> Python requirements
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pySlither SHARED pyWrapper/wrapper.cpp)
target_link_libraries(pySlither PRIVATE Slither)


