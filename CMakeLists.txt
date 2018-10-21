cmake_minimum_required(VERSION 3.8)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.33.tar.gz"
        SHA1 "acbf4b9b77b5a0160adc1da1cdda35f64b14196a"
        LOCAL
)
#include("cmake/config.cmake")
#set(HUNTER_KEEP_PACKAGE_SOURCES NO)

project(randomSlither)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)



#######################################
# Dependencies for the main library
#######################################
# Required :  Boost - Move python interface to pybind11
set(BUILD_SHARED_LIBS ON)
#add_definitions(-DBOOST_ALL_NO_LIB=1)
#set(Boost_USE_STATIC_LIBS OFF)
#set(BUILD_SHARED_LIBS)
hunter_add_package(Boost COMPONENTS program_options serialization)
find_package(Boost CONFIG REQUIRED program_options serialization)
#set_target_properties(Boost::serialization PROPERTIES POSITION_INDEPENDENT_CODE TRUE) # Adding -fPIC to boost

# Required : Opencv
find_package( OpenCV CONFIG REQUIRED)

# Optional OpenMP
find_package(OpenMP)

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


#######################################
# Build the library
#######################################
add_library(Slither SHARED ${SOURCE_FILES})

# Add related includes
target_include_directories(Slither PUBLIC lib)
target_include_directories(Slither PUBLIC source)
target_include_directories(Slither PUBLIC Boost::serialization)

# Link related libs
target_link_libraries(Slither PUBLIC ${OpenCV_LIBS})
target_link_libraries(Slither PUBLIC Boost::serialization)
target_link_libraries(Slither PUBLIC ${Boost_SERIALIZATION_LIBRARIES})
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
# To allow internal shared libs to work and pybind11 to be okay
#set_target_properties(Slither PROPERTIES POSITION_INDEPENDENT_CODE TRUE)



#######################################
# Build the Standalone executable
#######################################
add_executable(cppSlither source/main.cpp)
# Linking and including
target_link_libraries(cppSlither PUBLIC Slither)
target_link_libraries(cppSlither PUBLIC Boost::program_options)
target_link_libraries(Slither PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARIES})

set_target_properties(cppSlither PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )


#######################################
# Build the Python Wrapper
#######################################
# First --> Python requirements
#find_package(PythonInterp)
#find_package(PythonLibs)
hunter_add_package(pybind11)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pySlither SHARED pyWrapper/wrapper.cpp)
target_link_libraries(pySlither PUBLIC Slither)
#SET_TARGET_PROPERTIES(pySlither PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
#SET_TARGET_PROPERTIES(pySlither PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
#target_link_libraries(Slither PUBLIC ${Boost_SERIALIZATION_LIBRARIES})
#target_link_libraries(slither pybind11::pybind11 pybind11::embed pybind11::module)
#set_target_properties(slither PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#        SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::module).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
#set_target_properties(slither PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
#pybind11_add_module(slither SHARED NO_EXTRAS pyWrapper/wrapper.cpp)

#target_link_libraries(slither PUBLIC Slither)
#set_target_properties(pySlither PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


